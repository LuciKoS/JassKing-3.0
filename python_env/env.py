import sys
from typing import Tuple, Dict, Any  

import numpy as np
import random

from .cards import Rank
from . import cards as _cards_pkg

sys.modules.setdefault("cards", _cards_pkg)

from . import scoring as _scoring_pkg

sys.modules.setdefault("scoring", _scoring_pkg)

from .game_state import RoundState, PlayStatus
from .cards import decode_card

def one_hot(i: int, n: int) -> np.ndarray:
    v = np.zeros((n,), dtype=np.float32)
    if i >= 0 and i < n:
        v[i] = 1.0
    return v

class JassEnv:
    def __init__(self):
        self.state = RoundState()
        self.seed = 0
        self.trump = 0
        self.dealer = 0

    def reset(self, seed: int | None = None, trump: int | None = None, dealer: int | None = None) -> Tuple[tuple, Dict[str, Any]]:
        if seed is None:
            seed = random.randrange(1_000_000)
        self.seed = seed if seed is not None else self.seed
        if trump is not None:
            self.trump = trump
        if dealer is not None:
            self.dealer = dealer
        self.state.start_round(seed = self.seed, dealer = self.dealer, trump = self.trump)
        obs = self._encode_observation()
        mask = self._legal_action_mask()
        return obs, {"mask": mask}

    def zug(self, action: int) -> Tuple[tuple, float, bool, bool, Dict[str, Any]]:

        return self.reset()

    def _encode_observation(self) -> tuple:

        hist = np.zeros((4,9,36), dtype=np.float32)
        for t in range(min(self.state.round, 9)):
            for p in range(4):
                cid = self.state.cards[t][p]
                if cid != -1:
                    hist[p, t, cid] = 1.0

        ptr = np.zeros((4,1,36), dtype=np.float32)
        leader = self.state.leader
        for i, cid in enumerate (self.state.cards_on_table):
            pl = (leader + i) % 4
            ptr[pl, 0, cid] = 1.0

        hand_vec = np.zeros((36,), dtype=np.float32)
        hand_mask = self.state.hands_mask[self.state.current_player]
        tmp = hand_mask

        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            hand_vec[idx] = 1.0
            tmp ^= lsb

        mode_vec = one_hot(self.state.trump, 8)
        
        points_vec = np.array([self.state.points[0], self.state.points[1]], dtype=np.float32)

        turn_vec = one_hot(self.state.current_player, 4)

        rnd_vec = one_hot(min(self.state.round, 8), 9)

        geschoben_vec = np.zeros((2,), dtype=np.float32)

        return (hist, ptr, hand_vec, mode_vec, points_vec, turn_vec, rnd_vec, geschoben_vec)

    def _legal_action_mask(self) -> np.ndarray:

        mask = np.zeros((36,), dtype=np.float32)
        hand_mask = self.state.hands_mask[self.state.current_player]

        cards_in_hand = []
        tmp = hand_mask

        while tmp:
            lsb = tmp & -tmp
            idx = lsb.bit_length() - 1
            cards_in_hand.append(idx)
            tmp ^= lsb
        
        if not cards_in_hand:
            return mask

        if self.state.current_player == self.state.leader or self.state.suit_led == -1:
            for cid in cards_in_hand:
                mask[cid] = 1.0
            return mask

        suit_led = self.state.suit_led
        trump = self.state.trump

        in_suit = [cid for cid in cards_in_hand if decode_card(cid).suit == suit_led or decode_card(cid).suit == trump]
        if in_suit:
            if suit_led == trump:

                only_buur = (len(in_suit) == 1 and decode_card(in_suit[0]).rank == Rank.UNDER)
                if only_buur:
                    for cid in cards_in_hand:
                        mask[cid] = 1.0
                    return mask

            for cid in in_suit:
                mask[cid] = 1.0
            return mask

        for cid in cards_in_hand:
            mask[cid] = 1.0
        return mask

    def zug(self, action: int):

        mask = self._legal_action_mask()
        acting_seat = self.state.current_player
        acting_team = acting_seat % 2
        res = self.state.play_card(action)

        reward = 0.0

        if res.status.name in ("STICH_DONE", "ROUND_COMPLETE"):
            trick_idx = res.round_no
            stich_points = max(0, res.stich_punkte)
            winner_team = self.state.winners[trick_idx]

            if winner_team == acting_team:
                reward = float(stich_points)
            else:
                reward = -float(stich_points)


        done = (res.status.name == "ROUND_COMPLETE")
        truncated = False

        obs = self._encode_observation()
        mask = self._legal_action_mask()

        return obs, reward, done, truncated, {"mask": mask}

jassgame = JassEnv()