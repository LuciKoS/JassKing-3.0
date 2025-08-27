import random
from typing import Tuple, Dict, Any

import numpy as np

from python_env.game_state import RoundState, PlayStatus
from python_env.cards import decode_card


def one_hot(index: int, length: int) -> np.ndarray:
	v = np.zeros((length,), dtype=np.float32)
	if 0 <= index < length:
		v[index] = 1.0
	return v


class JassEnv:
	"""
	Minimal environment wrapper around RoundState to support DQN training.

	Observations match the DQN forward signature:
	- hist: (4, 9, 36)
	- ptr: (4, 1, 36)
	- hand: (36,)
	- mode: (8,)
	- points: (2,)
	- turn: (4,)
	- rnd: (9,)
	- geschoben: (2,)  # not used -> zeros
	"""

	def __init__(self):
		self.state = RoundState()
		self.seed = 0
		self.trump = 0  # 0..7
		self.dealer = 0
		# cache for last completed stich points to compute rewards cleanly
		self._last_round_index_reported = -1

	def reset(self, seed: int | None = None, trump: int | None = None, dealer: int | None = None) -> Tuple[tuple, Dict[str, Any]]:
		if seed is None:
			seed = random.randrange(1_000_000)
		self.seed = seed if seed is not None else self.seed
		if trump is not None:
			self.trump = trump
		if dealer is not None:
			self.dealer = dealer
		self.state.start_round(seed=self.seed, dealer=self.dealer, trump=self.trump)
		self._last_round_index_reported = -1
		obs = self._encode_observation()
		mask = self._legal_action_mask()
		return obs, {"mask": mask}

	def zug(self, action: int) -> Tuple[tuple, float, bool, bool, Dict[str, Any]]:
		# if illegal, small penalty and no state change beyond returning same encoded obs
		mask = self._legal_action_mask()
		reward = 0.0
		if action < 0 or action >= 36 or mask[action] == 0:
			# discourage illegal choices (should be masked by the trainer anyway)
			reward = -0.1
			obs = self._encode_observation()
			return obs, reward, False, False, {"mask": mask}

		res = self.state.play_card(action)

		# reward shaping: only when a stich completes, attribute points to acting player's team
		if res.status in (PlayStatus.STICH_DONE, PlayStatus.ROUND_COMPLETE):
			# points for the finished trick are stored at index round-1
			trick_idx = res.round_no
			stich_points = max(0, res.stich_punkte)
			# Determine which team won that trick: winners[trick_idx] is 0 for team0, 1 for team1
			winner_team = self.state.winners[trick_idx]
			acting_team = ((self.state.leaders[trick_idx]) % 2) if trick_idx >= 0 else 0
			# If the winner team equals the team of the leader at that trick, grant points; else 0
			reward = float(stich_points if winner_team == acting_team else 0.0)

		done = res.status == PlayStatus.ROUND_COMPLETE
		truncated = False

		obs = self._encode_observation()
		mask = self._legal_action_mask()
		return obs, reward, done, truncated, {"mask": mask}

	def _encode_observation(self) -> tuple:
		# hist: played cards per player per completed trick
		hist = np.zeros((4, 9, 36), dtype=np.float32)
		for t in range(min(self.state.round, 9)):
			for p in range(4):
				cid = self.state.cards[t][p]
				if cid != -1:
					hist[p, t, cid] = 1.0

		# ptr: current table composition, indexed by absolute player positions
		ptr = np.zeros((4, 1, 36), dtype=np.float32)
		leader = self.state.leader
		for i, cid in enumerate(self.state.cards_on_table):
			pl = (leader + i) % 4
			ptr[pl, 0, cid] = 1.0

		# hand vector: one-hot of current player's remaining cards
		hand_vec = np.zeros((36,), dtype=np.float32)
		hand_mask = self.state.hands_mask[self.state.current_player]
		tmp = hand_mask
		while tmp:
			lsb = tmp & -tmp
			idx = (lsb.bit_length() - 1)
			hand_vec[idx] = 1.0
			tmp ^= lsb

		# mode: trump 0..7 one-hot
		mode_vec = one_hot(self.state.trump, 8)

		# points: team totals so far
		points_vec = np.array([self.state.team_points[0], self.state.team_points[1]], dtype=np.float32)

		# turn: who is to act now (absolute 0..3)
		turn_vec = one_hot(self.state.current_player, 4)

		# rnd: trick index 0..8
		rnd_vec = one_hot(min(self.state.round, 8), 9)

		# geschoben: not modelled -> zeros length 2
		geschoben_vec = np.zeros((2,), dtype=np.float32)

		return (hist, ptr, hand_vec, mode_vec, points_vec, turn_vec, rnd_vec, geschoben_vec)

	def _legal_action_mask(self) -> np.ndarray:
		mask = np.zeros((36,), dtype=np.float32)
		hand_mask = self.state.hands_mask[self.state.current_player]
		# collect all cards in hand
		cards_in_hand = []
		tmp = hand_mask
		while tmp:
			lsb = tmp & -tmp
			idx = (lsb.bit_length() - 1)
			cards_in_hand.append(idx)
			tmp ^= lsb
		if not cards_in_hand:
			return mask  # no moves

		# if player leads, all cards in hand are legal
		if self.state.current_player == self.state.leader or self.state.suit_led == -1:
			for cid in cards_in_hand:
				mask[cid] = 1.0
			return mask

		# otherwise must follow suit if possible (with jack exception on trump suit)
		suit_led = self.state.suit_led
		trump = self.state.trump

		def suit_of(cid: int) -> int:
			return decode_card(cid).suit

		in_suit = [cid for cid in cards_in_hand if suit_of(cid) == suit_led]
		if in_suit:
			# special: only-trump-jack case when led suit equals trump
			if suit_led == trump:
				# count led-suit cards in hand (which are trump cards here)
				# If the only led-suit card is the Jack, player may discard any
				from python_env.cards import Rank  # local import to avoid cycle at top
				only_trump_jack = (len(in_suit) == 1 and decode_card(in_suit[0]).rank == Rank.JACK)
				if only_trump_jack:
					for cid in cards_in_hand:
						mask[cid] = 1.0
					return mask
			# otherwise must play any card of the led suit
			for cid in in_suit:
				mask[cid] = 1.0
			return mask
		# if cannot follow suit, any card is legal
		for cid in cards_in_hand:
			mask[cid] = 1.0
		return mask


# Export a singleton used by the training loop
jassgame = JassEnv()
