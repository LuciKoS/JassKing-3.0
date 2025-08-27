from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import torch
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from JassGott.agent import DQN, greedy_action, CFG

# --------------------------- Card system ---------------------------
SUITS = ["Schellen", "Schilten", "Eicheln", "Rosen"]  # Bells, Shields, Acorns, Roses
RANKS = ["6", "7", "8", "9", "B", "U", "O", "K", "A"]  # Banner=10, Under=J, Ober=Q, King, Ace
CARD_NAMES = [f"{s}-{r}" for s in SUITS for r in RANKS]  # 36 cards
CARD_SET = ["(empty)"] + CARD_NAMES

SUIT_OF = {name: i//9 for i, name in enumerate(CARD_NAMES)}
IDX_OF  = {name: i for i, name in enumerate(CARD_NAMES)}

MODES = [  # 8 modes -> one-hot length 8
    "Trumpf Schellen", "Trumpf Schilten", "Trumpf Eicheln", "Trumpf Rosen",
    "Obenabe", "Untenabe", "Slalom↑", "Slalom↓"
]

# --------------------------- Helper building ---------------------------

def one_hot(i: int, n: int):
    v = np.zeros((n,), dtype=np.float32)
    if 0 <= i < n:
        v[i] = 1.0
    return v

class History:
    """Tracks completed tricks for hist[4,9,36] and current trick for ptr[4,1,36]."""
    def __init__(self):
        self.reset_full()

    def reset_full(self):
        self.hist = np.zeros((4, 9, 36), dtype=np.float32)
        self.ptr  = np.zeros((4, 1, 36), dtype=np.float32)
        self.trick_idx = 0  # 0..8
        self.leader = 0     # player who leads current trick (0..3)
        self.turn = 0       # 0..3 relative to leader

    def add_card_to_ptr(self, player_abs: int, card_idx: int):
        self.ptr[player_abs, 0, :] = 0.0
        self.ptr[player_abs, 0, card_idx] = 1.0

    def complete_trick(self, table_idxs: list[int], leader_abs: int):
        # table_idxs length 4; order is play order from leader
        for t, card_idx in enumerate(table_idxs):
            pl = (leader_abs + t) % 4
            if card_idx is not None:
                self.hist[pl, self.trick_idx, card_idx] = 1.0
        self.trick_idx = min(self.trick_idx + 1, 8)
        self.ptr[:] = 0.0
        self.leader = (leader_abs + 1) % 4  # by default next leader -> set manually if needed
        self.turn = 0

# --------------------------- UI ---------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Jass ▶ DQN State Builder")
        self.geometry("980x640")
        self.resizable(True, True)

        self.history = History()
        self.games_done = 0

        self._build_left_hand()
        self._build_center_table()
        self._build_right_meta()
        self._build_bottom_actions()

    # ---- Left: 9 hand slots
    def _build_left_hand(self):
        f = ttk.LabelFrame(self, text="Hand (9 cards)")
        f.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.columnconfigure(0, weight=1)
        self.hand_vars = []
        for i in range(9):
            var = tk.StringVar(value="(empty)")
            cb = ttk.Combobox(f, textvariable=var, values=CARD_SET, state="readonly", width=22)
            cb.grid(row=i//3, column=i%3, padx=6, pady=6)
            self.hand_vars.append(var)

    # ---- Center: 4 table slots
    def _build_center_table(self):
        f = ttk.LabelFrame(self, text="Table (current trick: 4 plays)")
        f.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.columnconfigure(1, weight=1)
        self.table_vars = []
        for i in range(4):
            var = tk.StringVar(value="(empty)")
            cb = ttk.Combobox(f, textvariable=var, values=CARD_SET, state="readonly", width=22)
            cb.grid(row=0, column=i, padx=6, pady=6)
            self.table_vars.append(var)
        # leader & turn
        sub = ttk.Frame(f)
        sub.grid(row=1, column=0, columnspan=4, sticky="w", pady=(8,0))
        ttk.Label(sub, text="Leader (abs 0..3):").grid(row=0, column=0, padx=4)
        self.leader_var = tk.IntVar(value=0)
        self.turn_var = tk.IntVar(value=0)
        sp1 = ttk.Spinbox(sub, from_=0, to=3, textvariable=self.leader_var, width=4)
        sp1.grid(row=0, column=1, padx=4)
        ttk.Label(sub, text="Turn in trick (0..3):").grid(row=0, column=2, padx=8)
        sp2 = ttk.Spinbox(sub, from_=0, to=3, textvariable=self.turn_var, width=4)
        sp2.grid(row=0, column=3, padx=4)

    # ---- Right: meta info
    def _build_right_meta(self):
        f = ttk.LabelFrame(self, text="Meta")
        f.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        self.columnconfigure(2, weight=1)
        # mode
        ttk.Label(f, text="Mode (8):").grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value=MODES[0])
        self.mode_cb = ttk.Combobox(f, textvariable=self.mode_var, values=MODES, state="readonly", width=28)
        self.mode_cb.grid(row=1, column=0, pady=(0,8), sticky="w")
        # geschoben
        ttk.Label(f, text="Geschoben?").grid(row=2, column=0, sticky="w")
        self.geschoben_var = tk.IntVar(value=0)
        ttk.Radiobutton(f, text="Nein", variable=self.geschoben_var, value=0).grid(row=3, column=0, sticky="w")
        ttk.Radiobutton(f, text="Ja", variable=self.geschoben_var, value=1).grid(row=4, column=0, sticky="w")
        # points
        pts = ttk.Frame(f)
        pts.grid(row=5, column=0, sticky="w", pady=(10,0))
        ttk.Label(pts, text="Team A points:").grid(row=0, column=0, padx=4)
        ttk.Label(pts, text="Team B points:").grid(row=1, column=0, padx=4)
        self.pts_a = tk.IntVar(value=0)
        self.pts_b = tk.IntVar(value=0)
        ttk.Entry(pts, textvariable=self.pts_a, width=6).grid(row=0, column=1)
        ttk.Entry(pts, textvariable=self.pts_b, width=6).grid(row=1, column=1)
        # round index
        ttk.Label(f, text="Round (trick # 0..8)").grid(row=6, column=0, sticky="w", pady=(10,0))
        self.round_var = tk.IntVar(value=0)
        ttk.Spinbox(f, from_=0, to=8, textvariable=self.round_var, width=6).grid(row=7, column=0, sticky="w")
        # current player to act (absolute 0..3)
        ttk.Label(f, text="Player to act (abs 0..3)").grid(row=8, column=0, sticky="w", pady=(10,0))
        self.to_act_var = tk.IntVar(value=0)
        ttk.Spinbox(f, from_=0, to=3, textvariable=self.to_act_var, width=6).grid(row=9, column=0, sticky="w")

    # ---- Bottom: actions
    def _build_bottom_actions(self):
        f = ttk.Frame(self)
        f.grid(row=1, column=0, columnspan=3, sticky="ew", padx=10, pady=10)
        self.columnconfigure(0, weight=1)
        ttk.Button(f, text="Compute valid mask", command=self.on_compute_mask).grid(row=0, column=0, padx=6)
        ttk.Button(f, text="Add Trick to History", command=self.on_add_trick).grid(row=0, column=1, padx=6)
        ttk.Button(f, text="Greedy Action (load model)", command=self.on_greedy_action).grid(row=0, column=2, padx=6)
        ttk.Button(f, text="New Game", command=self.on_new_game).grid(row=0, column=3, padx=6)
        ttk.Button(f, text="Show State Shapes", command=self.on_show_shapes).grid(row=0, column=4, padx=6)

    # ------------------- State builders -------------------
    def hand_vector(self):
        v = np.zeros((36,), dtype=np.float32)
        for var in self.hand_vars:
            name = var.get()
            if name != "(empty)":
                v[IDX_OF[name]] = 1.0
        return v

    def ptr_tensor(self):
        # reflects current table entries
        ptr = np.zeros((4,1,36), dtype=np.float32)
        leader = self.leader_var.get()
        for i, var in enumerate(self.table_vars):
            name = var.get()
            if name != "(empty)":
                idx = IDX_OF[name]
                pl = (leader + i) % 4
                ptr[pl,0,idx] = 1.0
        return ptr

    def mode_vector(self):
        idx = MODES.index(self.mode_var.get())
        return one_hot(idx, 8)

    def points_vector(self):
        return np.array([self.pts_a.get(), self.pts_b.get()], dtype=np.float32)

    def turn_vector(self):
        return one_hot(self.to_act_var.get(), 4)

    def round_vector(self):
        return one_hot(self.round_var.get(), 9)

    def geschoben_vector(self):
        # [not pushed, pushed]
        return one_hot(self.geschoben_var.get(), 2)

    def get_state(self):
        hist = self.history.hist.copy()
        ptr  = self.ptr_tensor()
        hand = self.hand_vector()
        mode = self.mode_vector()
        pts  = self.points_vector()
        turn = self.turn_vector()
        rnd  = self.round_vector()
        geschoben = self.geschoben_vector()
        # shapes to match DQN forward: (4,9,36), (4,1,36), (36), (8), (2), (4), (9), (2)
        # convert to torch tensors when used with the model
        return (
            torch.from_numpy(hist),
            torch.from_numpy(ptr),
            torch.from_numpy(hand),
            torch.from_numpy(mode),
            torch.from_numpy(pts),
            torch.from_numpy(turn),
            torch.from_numpy(rnd),
            torch.from_numpy(geschoben),
        )

    # ------------------- Valid mask (simple follow-suit) -------------------
    def compute_valid_mask(self):
        mask = np.zeros((36,), dtype=np.float32)
        hand = self.hand_vector()
        on_table = [v.get() for v in self.table_vars if v.get() != "(empty)"]
        if not on_table:
            # any card in hand is ok
            mask = hand.copy()
            return torch.from_numpy(mask.astype(np.float32))
        lead_suit = SUIT_OF[on_table[0]]
        in_suit_indices = [i for i, n in enumerate(CARD_NAMES) if SUIT_OF[n] == lead_suit]
        has_in_suit = any(hand[i] > 0.5 for i in in_suit_indices)
        if has_in_suit:
            for i in in_suit_indices:
                if hand[i] > 0.5:
                    mask[i] = 1.0
        else:
            # can play anything you hold
            mask = hand.copy()
        return torch.from_numpy(mask.astype(np.float32))

    # ------------------- UI callbacks -------------------
    def on_compute_mask(self):
        m = self.compute_valid_mask()
        allowed = [CARD_NAMES[i] for i in range(36) if m[i] > 0.5]
        messagebox.showinfo("Valid moves", "\n".join(allowed) if allowed else "No valid moves (empty hand)")

    def on_add_trick(self):
        # collect current table
        table_names = [v.get() for v in self.table_vars]
        if "(empty)" in table_names:
            messagebox.showerror("Incomplete trick", "Please fill all 4 table slots before adding the trick.")
            return
        leader = self.leader_var.get()
        table_idxs = [IDX_OF[n] for n in table_names]
        self.history.complete_trick(table_idxs, leader)
        # clear table
        for v in self.table_vars:
            v.set("(empty)")
        # update round
        self.round_var.set(min(self.history.trick_idx, 8))
        # after finishing 9 tricks -> 1 game
        if self.history.trick_idx >= 9:
            self.games_done += 1
            if self.games_done % 3 == 0:
                self.history.reset_full()
                self.round_var.set(0)
                messagebox.showinfo("History reset", "3 games completed – history reset.")
            else:
                # start next game history
                self.history.reset_full()
                self.round_var.set(0)
                messagebox.showinfo("New game", f"Game {self.games_done} completed. Starting next game.")

    def on_show_shapes(self):
        state = self.get_state()
        shapes = [tuple(x.shape) for x in state]
        messagebox.showinfo("State shapes", str(shapes))

    def on_new_game(self):
        self.history.reset_full()
        self.round_var.set(0)
        for v in self.table_vars: v.set("(empty)")
        messagebox.showinfo("Reset", "History + current trick cleared.")

    def on_greedy_action(self):
        
        # Optional: let user select a checkpoint (can cancel)
        ckpt_path = filedialog.askopenfilename(title="Select model checkpoint (optional)", filetypes=[("PyTorch", "*.pt *.pth"), ("All files", "*.*")])
        device = CFG.device if torch.cuda.is_available() else "cpu"
        model = DQN(num_actions=CFG.num_actions).to(device)
        if ckpt_path:
            try:
                d = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(d.get('policy_state_dict', d))
            except Exception as e:
                messagebox.showwarning("Checkpoint load", f"Could not load checkpoint, using fresh weights.\n{e}")
        model.eval()
        state = self.get_state()
        valid_mask = self.compute_valid_mask()
        a = greedy_action(model, state, valid_mask, device)
        messagebox.showinfo("Greedy action", f"Action index (0..35): {a}\nCard: {CARD_NAMES[a]}")

# --------------------------- Run ---------------------------
if __name__ == "__main__":
    app = App()
    try:
        # use platform-appropriate theme if available
        from tkinter import ttk
        style = ttk.Style()
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    app.mainloop()
