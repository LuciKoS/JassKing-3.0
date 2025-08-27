from cards import Suit, Rank, Card, encode_card, decode_card, bit, add, remove
from scoring import winner, total_points
from typing import List
import random
from enum import Enum
from dataclasses import dataclass


class PlayStatus(Enum):
    OK = 0
    INVALID = 1
    STICH_DONE = 2
    ROUND_COMPLETE = 3

@dataclass
class PlayResult:
    status: PlayStatus
    reason: str = ""
    round_no: int = -1
    winner_seat: int = -1
    stich_punkte: int = -1

class RoundState:
    hands_mask: List[int]
    played_mask: int
    current_player: int
    trump: int
    dealer: int
    team_points: List[int]
    bonuses_by_team: List[int]
    round: int

    ##dynamic pro Stich
    leader: int
    cards_on_table: list[int]
    suit_led: int

    ##logging
    leaders: List[int]
    cards: List[List[int]]
    winners: List[int]
    points: List[int]


    def team_of(seat: int):
        return seat%2

    def next_player(self, seat: int):
        return (seat+1)%4

    def remaining_cards(self, seat: int):
        MASK_36 = (1 << 36) - 1
        return self.hands_mask[seat] & (~self.played_mask&MASK_36)
    



    

    def play_card(self, card_id: int):


        card = decode_card(card_id)
        hand = self.hands_mask[self.current_player]

        if not ((hand) >> card_id) & 1:
            return PlayResult(PlayStatus.INVALID, f"Player does not have this card {card}", self.round)
        
        if not self.current_player == self.leader:

            block_led = (hand >> (9 * self.suit_led)) & 0x1FF
            has_led = block_led != 0
            only_trump_jack = (self.suit_led == self.trump) and (block_led == (1 << Rank.UNDER))


            if card.suit != self.trump and card.suit != self.suit_led and has_led and not only_trump_jack:
                return PlayResult(PlayStatus.INVALID, f"Invalid Card {card}: must play card of correct suit", self.round)
        else:
            self.suit_led = card.suit


       
        self.cards_on_table.append(card_id)
        hand = remove(hand, card_id)
        self.cards[self.round][self.current_player] = card_id
        self.hands_mask[self.current_player] = hand
        self.current_player = self.next_player(self.current_player)

        if len(self.cards_on_table) == 4:
            self.end_stich()
            if self.round == 9:
                return PlayResult(PlayStatus.ROUND_COMPLETE, "", round_no=self.round -1, winner_seat=self.leader, stich_punkte=self.points[self.round-1])
            else:
                return PlayResult(PlayStatus.STICH_DONE, "", round_no=self.round -1, winner_seat=self.leader, stich_punkte=self.points[self.round-1])
        else:
            return PlayResult(PlayStatus.OK)
            
        

    def is_obe_abe(self):
        obeabe = False

        if (self.trump == 7):
            obeabe = True if (self.round%2) == 0 else False
        elif (self.trump == 6):
            obeabe = False if (self.round%2) == 1 else False
        elif (self.trump == 5):
            obeabe = True

        return obeabe
    
    def end_stich(self):

        winner_seat = winner(self.suit_led, self.cards_on_table, self.leader, self.trump, self.is_obe_abe)
        total = total_points(self.suit_led, self.cards_on_table, self.leader, self.trump)

        self.team_points[0 if (winner_seat%2) == 0 else 1] += total

        self.winners[self.round] = 0 if (winner_seat%2) == 0 else 1
        self.points[self.round] = total

        self.leader = winner_seat
        self.current_player = winner_seat
        self.round += 1
        self.cards_on_table.clear()

        if not self.round == 9:
            
            self.leaders[self.round] = self.leader
            self.suit_led = -1


    def start_round(self, seed: int, dealer: int = 0, trump: int = 0):
        self.dealer = dealer
        self.played_mask = 0
        self.cards_on_table.clear()
        self.current_player = self.next_player(dealer)
        self.leader = self.current_player
        self.suit_led = -1
        self.team_points = [0,0]
        self.bonuses_by_team = [0,0]
        self.trump = trump
        self.round = 0

        self.leaders[:] = [-1]*9
        for t in range(9):
            self.cards[t][:] = [-1,-1,-1,-1]
        self.winners[:] = [-1]*9
        self.points[:] = [-1]*9

        self.leaders[0] = self.leader

        rng = random.Random(seed)

        deck = list(range(36))
        rng.shuffle(deck)

        self.hands_mask = [0,0,0,0]

        for p in range(4):
            for k in range(9):
                self.hands_mask[p] = add(self.hands_mask[p], deck[p*9 + k])
                assert((p*9 + k) < 36)

        
        MASK_36 = (1 << 36) - 1

        # hands must be disjoint, within [0..35], and each have 9 cards
        union = 0
        for p, m in enumerate(self.hands_mask):
            assert (m & ~MASK_36) == 0, f"Player {p} mask has bits >=36 set"
            assert m.bit_count() == 9, f"Player {p} does not have 9 cards"
            assert (union & m) == 0, "Hands overlap!"
            union |= m
        assert union == MASK_36, "Not all 36 cards dealt"
    
    def __init__(self):
        self.hands_mask = [0,0,0,0]
        self.played_mask = 0
        self.current_player = 0
        self.trump = 0
        self.dealer = 0
        self.team_points = [0,0]
        self.bonuses_by_team = [0,0]
        self.round = 0
        self.leader = 0
        self.cards_on_table = []      # <-- initialize!
        self.suit_led = -1
        self.leaders = [-1]*9
        self.cards = [[-1]*4 for _ in range(9)]
        self.winners = [-1]*9
        self.points = [0]*9

