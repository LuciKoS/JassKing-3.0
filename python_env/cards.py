"""
Card system for 36 Jasskarten:
    Definition of the Cards ("Suit and Rank")
    How to represent cards in memory (efficient IDs and masks instead of slow Python lists) ?? (learn)
    Helper functions

    Cards: Suit & Rank 
        ->  encode_card(suit, rank) -> cardid
        -> decode_card(cardid) -> suit, rank
"""


"""Trumpf:
0 rose
1 eichel
2 schilte
3 schelle
4 une ufe
5 obe abe
6 une ufe slalom
7 obe abe slalom
"""

from enum import IntEnum
from dataclasses import dataclass

class Suit(IntEnum):
    ROSE = 0
    EICHEL = 1
    SCHILTE = 2
    SCHELLE = 3

class Rank(IntEnum):
    SIX = 0
    SEVEN = 1
    EIGHT = 2
    NINE = 3
    TEN = 4
    UNDER = 5
    OBER = 6
    KOENIG = 7
    ASS = 8

@dataclass(frozen = True)
class Card:
    suit: Suit
    rank: Rank
    id: int

    def __str__(self) -> str:
        return f"{self.rank.name.title()} of {self.suit.name.title()}"

CARD_BY_ID = [None]*36
ID_BY_SUIT_RANK = [[None]*9 for _ in range(4)]
SUIT_MASKS = [0] * 4
RANK_MASKS = [0] * 9
DECK_MASK = 0

def _init_tables():
    global CARD_BY_ID, ID_BY_SUIT_RANK, SUIT_MASKS, RANK_MASKS, DECK_MASK
    card_id = 0
    for suit in range(4):
        for rank in range(9):
            try:
                card_obj = Card(Suit(suit), Rank(rank), card_id)
            except Exception:
                card_obj = (suit, rank, card_id)
            
            CARD_BY_ID[card_id] = card_obj
            ID_BY_SUIT_RANK[suit][rank] = card_id

            SUIT_MASKS[suit] |= 1 << card_id
            RANK_MASKS[rank] |= 1 << card_id

            DECK_MASK |= 1 << card_id

            card_id += 1

_init_tables()

"""
Card things
"""

def encode_card(suit: Suit, rank: Rank) -> int:
    return 9*int(suit) + int(rank)

def decode_card(id) -> Card:
    rank = id % 9
    suit = id // 9
    return Card(Suit(suit), Rank(rank), id)

"""
Bitmasks to represent cards
"""

"return mask with only one card"
def bit(card_id: int) -> int:
    mask = 1 << card_id
    return mask

"adds card to existing mask"
def add(mask: int, card_id: int):
    return mask | (1 << card_id)

def remove(mask: int, card_id: int):
    return mask & ~(1 << card_id)