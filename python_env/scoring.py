from cards import Suit, Rank, Card, encode_card, decode_card, bit, add, remove

NON_TRUMP_SCORE = [0, 0, 0, 0, 10, 2, 3, 4, 11]
TRUMP_SCORE     = [0, 0, 0, 14, 10, 20, 3, 4, 11]
OBE_ABE         = [0,0,8,0,10,2,3,4,11]
UNE_UFE         = [11,0,8,0,10,2,3,4,0]



def total_points(suit_led: int, cards_on_table: list[int], leader: int, trump: int):
    
    sum = 0
    
    if trump < 4:
        for card_id in cards_on_table:
            card = decode_card(card_id)
            if (card.suit == trump):
                sum += TRUMP_SCORE[card.rank]
            else:
                sum += NON_TRUMP_SCORE[card.rank]

        if trump > 1:
            sum *= 2
    
    elif trump == 4 or trump == 6:
        for card_id in cards_on_table:
            card = decode_card(card_id)
            sum += UNE_UFE[card.rank]
        
        sum *= 3 if trump == 4 else 4
    elif trump == 5 or trump == 7:
        for card_id in cards_on_table:
            card = decode_card(card_id)
            sum += OBE_ABE[card.rank]
        sum *= 3 if trump == 5 else 4

    return sum


NON_TRUMP_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8]
TRUMP_ORDER     = [0, 1, 2, 7, 3, 8, 4, 5, 6]   
UNE_UFE         = [8, 7, 6, 5, 4, 3, 2, 1, 0]
OBE_ABE         = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    
def winner(suit_led: int, cards_on_table: list[int], leader: int, trump: int, obe_abe: bool):
    
    best_strength = -1
    winner_seat = leader

    for i, card_id in enumerate(cards_on_table):
        seat = (leader + i)% 4
        card = decode_card(card_id)

        if trump < 4:
            if card.suit == trump:
                s = 200 + TRUMP_ORDER[card.rank]
            elif card.suit == suit_led:
                s = 100 + NON_TRUMP_ORDER[card.rank]
            else:
                s = -1
        elif obe_abe:
            if card.suit == suit_led:
                s = OBE_ABE[card.rank]
            else:
                s = -1
        else:
            if card.suit == suit_led:
                s = UNE_UFE[card.rank]
            else:
                s = -1

        if s > best_strength:
            winner_seat = seat
            best_strength = s
    
    
    
    return winner_seat 


    




