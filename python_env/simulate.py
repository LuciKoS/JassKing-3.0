from game_state import RoundState, PlayResult, PlayStatus
from cards import Suit, Rank, Card, encode_card, decode_card, bit, add, remove

from enum import Enum
from dataclasses import dataclass
import random



def cid_to_compact(cid: int) -> str:
    card = decode_card(cid)
    suit_char = card.suit.name[0]  # R/E/S/S (Rose/Eichel/Schilte/Schelle)
    rank_map = {
        Rank.SIX: "6", Rank.SEVEN: "7", Rank.EIGHT: "8",
        Rank.NINE: "9", Rank.TEN: "10", Rank.JACK: "U",
        Rank.QUEEN: "O", Rank.KING: "K", Rank.ACE: "A",
    }
    return f"{suit_char}{rank_map[card.rank]}"

def hand_to_list(mask: int) -> list[int]:
    cards = []
    while mask:
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        cards.append(idx)
        mask ^= lsb
    return cards

def hand_to_string(mask: int) -> str:
    return " ".join(cid_to_compact(c) for c in hand_to_list(mask))

def parse_card(spec: str):
    spec = spec.strip().upper()
    if not spec:
        raise ValueError("Empty card string")
    if spec.isdigit():
        cid = int(spec)
        if 0 <= cid < 36:
            return cid
        raise ValueError("Card id must be 0..35")

    suit_map = {"R": Suit.ROSE, "E": Suit.EICHEL, "S": Suit.SCHILTE, "C": Suit.SCHELLE}
    rank_map = {"6": Rank.SIX, "7": Rank.SEVEN, "8": Rank.EIGHT, "9": Rank.NINE,
                "10": Rank.TEN, "U": Rank.JACK, "O": Rank.QUEEN, "K": Rank.KING, "A": Rank.ACE}

    s = spec[0]
    r = spec[1:]
    if s in suit_map and r in rank_map:
        return encode_card(suit_map[s], rank_map[r])

    raise ValueError("Card format e.g. R10, EJ, S6, CA, or numeric id 0..35")





def render_state(state: RoundState):
    round = state.round
    stich_str = f"{round + 1}"
    trump_str = f"{state.trump}"

    print("\n=== Jass Simulator ===")
    print(f"Trump: {trump_str}    Trick: {stich_str}")
    print(f"Scores  Team0:{state.team_points[0]}  Team1:{state.team_points[1]}\n")

    tag2 = "-> " if state.current_player == 2 else "   "
    print(f"{tag2}P2: {hand_to_string(state.hands_mask[2])}\n")

    tag3 = "-> " if state.current_player == 3 else "   "
    tag1 = "-> " if state.current_player == 1 else "   "
    print(f"{tag3}P3: {hand_to_string(state.hands_mask[3])}")
    table_cards = " ".join(cid_to_compact(c) for c in state.cards_on_table)
    print(f"Table (led={getattr(state,'suit_led','-')}): {table_cards}")
    print(f"{tag1}P1: {hand_to_string(state.hands_mask[1])}\n")

    tag0 = "-> " if state.current_player == 0 else "   "
    print(f"{tag0}P0: {hand_to_string(state.hands_mask[0])}\n")

def render_stich_summary(state: RoundState, res: PlayResult):
    plays = " ".join(cid_to_compact(c) for c in state.cards_on_table)
    winner = res.winner_seat
    pts = res.stich_punkte
    print("Stich complete.")
    if plays:
        print(f"Plays: {plays}")
    if winner != -1:
        print(f"Winner: P{winner} (Team {winner % 2})")
    if pts != -1:
        print(f"Points this stich: {pts}")
    print(f"Totals -> Team0:{state.team_points[0]}  Team1:{state.team_points[1]}")


class Simulator:
    def __init__(self, seed: int = 123, trump: int = 4, dealer: int = 0):
        self.seed = seed
        self.trump = trump
        self.dealer = dealer
        self.state = RoundState()
        self.restart(seed, trump, dealer)

    def restart(self, seed: int | None = None, trump: int | None = None, dealer: int | None = None):
        if seed is not None:
            self.seed = seed
        if trump is not None:
            self.trump = trump
        if dealer is not None:
            self.dealer = dealer
        self.state.start_round(seed=self.seed, dealer=self.dealer, trump=self.trump)
        render_state(self.state)

    def handle_play(self, token: str):
        try:
            cid = parse_card(token)
        except Exception as e:
            print(f"Parse error: {e}")
            return
        
        res: PlayResult = self.state.play_card(cid)

        if res.status == PlayStatus.INVALID:
            reason = getattr(res, "reason", "Invalid")
            rnd = getattr(res, "round", None)
            print(f"INVALID: {reason}{'' if rnd is None else f' (trick {rnd+1})'}")
            return

        if res.status == PlayStatus.OK:
            # legal, trick not finished
            render_state(self.state)
            return

        if res.status == PlayStatus.STICH_DONE:
            # show summary, then the next-trick state
            render_stich_summary(self.state, res)
            render_state(self.state)
            return
        
        if res.status == PlayStatus.ROUND_COMPLETE:
            render_stich_summary(self.state, res)
            print("ROUND COMPLETE.")
            print(f"Final -> Team0:{self.state.team_points[0]}  Team1:{self.state.team_points[1]}")
            return
        
        render_state(self.state)


    def run(self):
        print("Commands: deal [seed [trump [dealer]]] | play <card> | state | stich | help | quit")

        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye")
                break

            if not line: 
                continue

            if line in ("quit", "exit"):
                break

            if line == "help":
                print("Commands:")
                print("  deal [seed [trump [dealer]]]   restart with given parameters")
                print("  play <card>                    play next card (e.g., R10, EJ, 23)")
                print("  state                          show current state")
                print("  trick                          show current trick cards")
                print("  quit                           exit")
                continue

            if line == "state":
                render_state(self.state)
                continue

            if line == "stich":
                print("Current stich:", " ".join(cid_to_compact(c) for c in self.state.cards_on_table))
                continue

            if line.startswith("deal"):
                parts = line.split()
                # seed default: randomize if omitted; keep current trump/dealer if omitted
                seed = int(parts[1]) if len(parts) > 1 else random.randrange(1_000_000)
                trump = int(parts[2]) if len(parts) > 2 else self.trump
                dealer = int(parts[3]) if len(parts) > 3 else self.dealer
                self.restart(seed=seed, trump=trump, dealer=dealer)
                continue

            if line.startswith("play "):
                token = line.split(maxsplit=1)[1]
                self.handle_play(token)
                continue

            print("Unknown command. Type 'help'.")


if __name__ == "__main__":
    Simulator().run()