from __future__ import annotations
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
import numpy as np
import torch
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from JassGott.agent import DQN, greedy_action, CFG

def load_policy(checkpoint_path):
    device = CFG.device
    model = DQN(num_actions=CFG.num_actions).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['policy_state_dict'])
    model.eval()
    return model, device

# --------------------------- Card system ---------------------------
SUITS = ["Rosen", "Eicheln", "Schilten", "Schellen"]
RANKS = ["6", "7", "8", "9", "10", "U", "O", "K", "A"]
CARD_NAMES = [f"{s}-{r}" for s in SUITS for r in RANKS]

SUIT_OF = {name: i//9 for i, name in enumerate(CARD_NAMES)}
IDX_OF  = {name: i for i, name in enumerate(CARD_NAMES)}

MODES = [
    "Trumpf Schellen", "Trumpf Schilten", "Trumpf Eicheln", "Trumpf Rosen",
    "Unteufe", "Obeabe", "Slalom↑", "Slalom↓"
]

# --------------------------- Helper functions ---------------------------
def one_hot(i: int, n: int):
    v = np.zeros((n,), dtype=np.float32)
    if 0 <= i < n:
        v[i] = 1.0
    return v

class History:
    def __init__(self):
        self.reset_full()

    def reset_full(self):
        self.hist = np.zeros((4, 9, 36), dtype=np.float32)
        self.ptr  = np.zeros((4, 1, 36), dtype=np.float32)
        self.trick_idx = 0
        self.leader = 0
        self.turn = 0

    def add_card_to_ptr(self, player_abs: int, card_idx: int):
        self.ptr[player_abs, 0, :] = 0.0
        self.ptr[player_abs, 0, card_idx] = 1.0

    def complete_trick(self, table_idxs: list[int], leader_abs: int):
        for t, card_idx in enumerate(table_idxs):
            pl = (leader_abs + t) % 4
            if card_idx is not None:
                self.hist[pl, self.trick_idx, card_idx] = 1.0
        self.trick_idx = min(self.trick_idx + 1, 8)
        self.ptr[:] = 0.0
        self.leader = (leader_abs + 1) % 4
        self.turn = 0

class ImageCardButton(tk.Button):
    def __init__(self, parent, card_idx, image, **kwargs):
        super().__init__(parent, image=image, **kwargs)
        self.card_idx = card_idx
        self.selected = False
        self.update_appearance()
    
    def update_appearance(self):
        if self.selected:
            self.config(relief='sunken', borderwidth=4, bg='#4CAF50', fg='white')
        else:
            self.config(relief='raised', borderwidth=2, bg='#f0f0f0', fg='black')

# --------------------------- Beautiful UI ---------------------------
class JassTestApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🎴 Jass AI Model Tester")
        self.geometry("1600x1000")
        self.configure(bg='#2C3E50')
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Title.TLabel', font=('Arial', 16, 'bold'), foreground='#ECF0F1')
        self.style.configure('Header.TLabel', font=('Arial', 12, 'bold'), foreground='#ECF0F1')
        self.style.configure('Card.TFrame', background='#34495E', relief='raised', borderwidth=2)
        self.style.configure('Action.TButton', font=('Arial', 10, 'bold'), padding=10)

        self.history = History()
        self.selected_hand_cards = set()
        self.selected_table_cards = [None] * 4
        self.current_player = 0
        self.leader = 0
        self.trump = 0
        
        self.model = None
        self.device = None
        
        # Load card images
        self.card_images = []
        self.load_card_images()
        
        self._build_interface()

    def load_card_images(self):
        """Load all card images from images3 folder"""
        images_dir = os.path.join(ROOT, "images3")
        for i in range(36):
            try:
                img_path = os.path.join(images_dir, f"img_{i}.jpg")
                img = Image.open(img_path)
                # Resize to nice size for buttons
                img = img.resize((70, 95), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self.card_images.append(photo)
            except Exception as e:
                print(f"Could not load image {i}: {e}")
                self.card_images.append(None)

    def _build_interface(self):
        # Main container
        main_container = tk.Frame(self, bg='#2C3E50')
        main_container.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Title
        title_frame = tk.Frame(main_container, bg='#2C3E50')
        title_frame.pack(fill='x', pady=(0, 20))
        
        title_label = tk.Label(title_frame, text="🎴 Jass AI Model Tester", 
                              font=('Arial', 24, 'bold'), 
                              bg='#2C3E50', fg='#ECF0F1')
        title_label.pack()
        
        subtitle_label = tk.Label(title_frame, text="Test your trained model in real game situations", 
                                 font=('Arial', 12), 
                                 bg='#2C3E50', fg='#BDC3C7')
        subtitle_label.pack()
        
        # Model loading section
        model_frame = tk.Frame(main_container, bg='#34495E', relief='raised', bd=2)
        model_frame.pack(fill='x', pady=(0, 20))
        
        model_inner = tk.Frame(model_frame, bg='#34495E', padx=20, pady=15)
        model_inner.pack(fill='x')
        
        tk.Label(model_inner, text="🤖 AI Model:", font=('Arial', 12, 'bold'), 
                bg='#34495E', fg='#ECF0F1').pack(side='left')
        
        self.model_path_var = tk.StringVar(value="checkpoints/15mil.pt")
        model_entry = tk.Entry(model_inner, textvariable=self.model_path_var, 
                              font=('Arial', 10), width=40, 
                              bg='#ECF0F1', fg='#2C3E50', relief='sunken', bd=2)
        model_entry.pack(side='left', padx=10)
        
        load_btn = tk.Button(model_inner, text="📁 Load Model", command=self.load_model,
                            font=('Arial', 10, 'bold'), bg='#3498DB', fg='white',
                            relief='raised', bd=2, padx=15, pady=5)
        load_btn.pack(side='left', padx=5)
        
        rec_btn = tk.Button(model_inner, text="�� Get Recommendation", command=self.get_recommendation,
                           font=('Arial', 10, 'bold'), bg='#E74C3C', fg='white',
                           relief='raised', bd=2, padx=15, pady=5)
        rec_btn.pack(side='left', padx=5)
        
        # Main content area
        content_frame = tk.Frame(main_container, bg='#2C3E50')
        content_frame.pack(fill='both', expand=True)
        
        # Left: Hand selection
        hand_frame = tk.Frame(content_frame, bg='#34495E', relief='raised', bd=2)
        hand_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        hand_header = tk.Frame(hand_frame, bg='#34495E')
        hand_header.pack(fill='x', padx=15, pady=10)
        
        tk.Label(hand_header, text="🃏 Your Hand", font=('Arial', 14, 'bold'), 
                bg='#34495E', fg='#ECF0F1').pack(side='left')
        
        hand_count_label = tk.Label(hand_header, text="(0/9 cards selected)", 
                                   font=('Arial', 10), bg='#34495E', fg='#BDC3C7')
        hand_count_label.pack(side='right')
        self.hand_count_label = hand_count_label
        
        # Hand cards with scroll
        hand_canvas_frame = tk.Frame(hand_frame, bg='#34495E')
        hand_canvas_frame.pack(fill='both', expand=True, padx=15, pady=15)
        
        hand_canvas = tk.Canvas(hand_canvas_frame, bg='#34495E', highlightthickness=0)
        hand_scrollbar = ttk.Scrollbar(hand_canvas_frame, orient="vertical", command=hand_canvas.yview)
        hand_scrollable_frame = tk.Frame(hand_canvas, bg='#34495E')
        
        hand_scrollable_frame.bind(
            "<Configure>",
            lambda e: hand_canvas.configure(scrollregion=hand_canvas.bbox("all"))
        )
        
        hand_canvas.create_window((0, 0), window=hand_scrollable_frame, anchor="nw")
        hand_canvas.configure(yscrollcommand=hand_scrollbar.set)
        
        # Create hand buttons in a grid
        self.hand_buttons = []
        for i in range(36):
            if self.card_images[i]:
                btn = ImageCardButton(hand_scrollable_frame, i, self.card_images[i],
                                    command=lambda idx=i: self.toggle_hand_card(idx))
            else:
                btn = ImageCardButton(hand_scrollable_frame, i, None,
                                    text=f"Card {i}", width=10, height=4,
                                    command=lambda idx=i: self.toggle_hand_card(idx))
            btn.grid(row=i//6, column=i%6, padx=3, pady=3)
            self.hand_buttons.append(btn)
        
        hand_canvas.pack(side="left", fill="both", expand=True)
        hand_scrollbar.pack(side="right", fill="y")
        
        # Center: Game state
        center_frame = tk.Frame(content_frame, bg='#2C3E50')
        center_frame.pack(side='left', fill='both', expand=True, padx=10)
        
        # Game info
        info_frame = tk.Frame(center_frame, bg='#34495E', relief='raised', bd=2)
        info_frame.pack(fill='x', pady=(0, 15))
        
        info_inner = tk.Frame(info_frame, bg='#34495E', padx=20, pady=15)
        info_inner.pack(fill='x')
        
        tk.Label(info_inner, text="�� Game State", font=('Arial', 14, 'bold'), 
                bg='#34495E', fg='#ECF0F1').pack(anchor='w', pady=(0, 10))
        
        # Game controls in a grid
        controls_frame = tk.Frame(info_inner, bg='#34495E')
        controls_frame.pack(fill='x')
        
        # Row 1
        tk.Label(controls_frame, text="🎯 Trump:", font=('Arial', 10, 'bold'), 
                bg='#34495E', fg='#ECF0F1').grid(row=0, column=0, padx=(0, 10), pady=5, sticky='w')
        
        self.trump_var = tk.StringVar(value=MODES[0])
        trump_combo = ttk.Combobox(controls_frame, textvariable=self.trump_var, values=MODES, 
                                  state="readonly", width=20, font=('Arial', 10))
        trump_combo.grid(row=0, column=1, padx=(0, 20), pady=5)
        trump_combo.bind('<<ComboboxSelected>>', self.update_trump)
        
        tk.Label(controls_frame, text="👤 Current Player:", font=('Arial', 10, 'bold'), 
                bg='#34495E', fg='#ECF0F1').grid(row=0, column=2, padx=(0, 10), pady=5, sticky='w')
        
        self.player_var = tk.IntVar(value=0)
        ttk.Spinbox(controls_frame, from_=0, to=3, textvariable=self.player_var, 
                   width=4, font=('Arial', 10)).grid(row=0, column=3, padx=(0, 20), pady=5)
        
        # Row 2
        tk.Label(controls_frame, text="�� Leader:", font=('Arial', 10, 'bold'), 
                bg='#34495E', fg='#ECF0F1').grid(row=1, column=0, padx=(0, 10), pady=5, sticky='w')
        
        self.leader_var = tk.IntVar(value=0)
        ttk.Spinbox(controls_frame, from_=0, to=3, textvariable=self.leader_var, 
                   width=4, font=('Arial', 10)).grid(row=1, column=1, padx=(0, 20), pady=5, sticky='w')
        
        tk.Label(controls_frame, text="�� Trick #:", font=('Arial', 10, 'bold'), 
                bg='#34495E', fg='#ECF0F1').grid(row=1, column=2, padx=(0, 10), pady=5, sticky='w')
        
        self.trick_var = tk.IntVar(value=0)
        ttk.Spinbox(controls_frame, from_=0, to=8, textvariable=self.trick_var, 
                   width=4, font=('Arial', 10)).grid(row=1, column=3, padx=(0, 20), pady=5, sticky='w')
        
        # Table (current trick)
        table_frame = tk.Frame(center_frame, bg='#34495E', relief='raised', bd=2)
        table_frame.pack(fill='both', expand=True)
        
        table_header = tk.Frame(table_frame, bg='#34495E', padx=15, pady=10)
        table_header.pack(fill='x')
        
        tk.Label(table_header, text="🃏 Current Trick", font=('Arial', 14, 'bold'), 
                bg='#34495E', fg='#ECF0F1').pack(side='left')
        
        table_inner = tk.Frame(table_frame, bg='#34495E', padx=15, pady=15)
        table_inner.pack(fill='both', expand=True)
        
        self.table_buttons = []
        for i in range(4):
            btn = tk.Button(table_inner, text=f"Player {i}\n(empty)", 
                           font=('Arial', 10, 'bold'), width=12, height=6,
                           bg='#ECF0F1', fg='#2C3E50', relief='raised', bd=2,
                           command=lambda pos=i: self.select_table_card(pos))
            btn.grid(row=0, column=i, padx=10, pady=10)
            self.table_buttons.append(btn)
        
        # Right: Recommendation and actions
        right_frame = tk.Frame(content_frame, bg='#2C3E50')
        right_frame.pack(side='right', fill='y', padx=(10, 0))
        
        # Recommendation display
        rec_frame = tk.Frame(right_frame, bg='#34495E', relief='raised', bd=2)
        rec_frame.pack(fill='x', pady=(0, 15))
        
        rec_header = tk.Frame(rec_frame, bg='#34495E', padx=15, pady=10)
        rec_header.pack(fill='x')
        
        tk.Label(rec_header, text="🎯 AI Recommendation", font=('Arial', 14, 'bold'), 
                bg='#34495E', fg='#ECF0F1').pack()
        
        rec_content = tk.Frame(rec_frame, bg='#34495E', padx=15, pady=15)
        rec_content.pack(fill='x')
        
        self.rec_label = tk.Label(rec_content, text="Load model first", 
                                 font=('Arial', 12), bg='#34495E', fg='#BDC3C7',
                                 wraplength=200)
        self.rec_label.pack(pady=10)
        
        # Actions
        action_frame = tk.Frame(right_frame, bg='#34495E', relief='raised', bd=2)
        action_frame.pack(fill='x')
        
        action_header = tk.Frame(action_frame, bg='#34495E', padx=15, pady=10)
        action_header.pack(fill='x')
        
        tk.Label(action_header, text="⚡ Actions", font=('Arial', 14, 'bold'), 
                bg='#34495E', fg='#ECF0F1').pack()
        
        action_content = tk.Frame(action_frame, bg='#34495E', padx=15, pady=15)
        action_content.pack(fill='x')
        
        clear_btn = tk.Button(action_content, text="🗑️ Clear All", command=self.clear_all,
                             font=('Arial', 10, 'bold'), bg='#E67E22', fg='white',
                             relief='raised', bd=2, padx=20, pady=8)
        clear_btn.pack(fill='x', pady=2)
        
        complete_btn = tk.Button(action_content, text="✅ Complete Trick", command=self.complete_trick,
                                font=('Arial', 10, 'bold'), bg='#27AE60', fg='white',
                                relief='raised', bd=2, padx=20, pady=8)
        complete_btn.pack(fill='x', pady=2)
        
        new_game_btn = tk.Button(action_content, text="�� New Game", command=self.new_game,
                                font=('Arial', 10, 'bold'), bg='#9B59B6', fg='white',
                                relief='raised', bd=2, padx=20, pady=8)
        new_game_btn.pack(fill='x', pady=2)

    def toggle_hand_card(self, card_idx):
        if card_idx in self.selected_hand_cards:
            self.selected_hand_cards.remove(card_idx)
            self.hand_buttons[card_idx].selected = False
        else:
            if len(self.selected_hand_cards) < 9:  # Max 9 cards in hand
                self.selected_hand_cards.add(card_idx)
                self.hand_buttons[card_idx].selected = True
        self.hand_buttons[card_idx].update_appearance()
        self.update_hand_count()

    def update_hand_count(self):
        count = len(self.selected_hand_cards)
        self.hand_count_label.config(text=f"({count}/9 cards selected)")

    def select_table_card(self, position):
        dialog = CardSelectionDialog(self, "Select card for table position")
        if dialog.result is not None:
            self.selected_table_cards[position] = dialog.result
            if self.card_images[dialog.result]:
                self.table_buttons[position].config(image=self.card_images[dialog.result])
            else:
                self.table_buttons[position].config(text=f"Player {position}\n{CARD_NAMES[dialog.result]}")

    def load_model(self):
        try:
            checkpoint_path = self.model_path_var.get()
            self.model, self.device = load_policy(checkpoint_path)
            messagebox.showinfo("✅ Success", f"Model loaded from {checkpoint_path}")
        except Exception as e:
            messagebox.showerror("❌ Error", f"Could not load model: {e}")

    def get_recommendation(self):
        if self.model is None:
            messagebox.showwarning("⚠️ Warning", "Please load a model first")
            return
        
        try:
            state = self.get_state()
            valid_mask = self.compute_valid_mask()
            
            # Convert to torch tensors
            state_t = tuple(torch.as_tensor(x).to(self.device).float() for x in state)
            mask_t = torch.as_tensor(valid_mask, dtype=torch.bool).to(self.device)
            
            # Get greedy action
            action = greedy_action(self.model, state_t, mask_t, self.device)
            
            # Update recommendation display
            if self.card_images[action]:
                # Create a small recommendation image
                rec_img = self.card_images[action]
                self.rec_label.config(image=rec_img, text="")
            else:
                self.rec_label.config(image="", text=f"Recommended:\n{CARD_NAMES[action]}\nIndex: {action}")
            
        except Exception as e:
            messagebox.showerror("❌ Error", f"Could not get recommendation: {e}")

    def get_state(self):
        hist = self.history.hist.copy()
        ptr = np.zeros((4, 1, 36), dtype=np.float32)
        
        # Add current table cards to ptr
        for i, card_idx in enumerate(self.selected_table_cards):
            if card_idx is not None:
                player = (self.leader_var.get() + i) % 4
                ptr[player, 0, card_idx] = 1.0
        
        # Build hand vector
        hand = np.zeros((36,), dtype=np.float32)
        for card_idx in self.selected_hand_cards:
            hand[card_idx] = 1.0
        
        # Other vectors
        trump_idx = MODES.index(self.trump_var.get())
        mode = one_hot(trump_idx, 8)
        pts = np.array([0, 0], dtype=np.float32)
        turn = one_hot(self.player_var.get(), 4)
        rnd = one_hot(self.trick_var.get(), 9)
        geschoben = np.zeros((2,), dtype=np.float32)
        
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

    def compute_valid_mask(self):
        mask = np.zeros((36,), dtype=np.float32)
        hand = np.zeros((36,), dtype=np.float32)
        for card_idx in self.selected_hand_cards:
            hand[card_idx] = 1.0
        
        table_cards = [c for c in self.selected_table_cards if c is not None]
        if not table_cards:
            mask = hand.copy()
        else:
            lead_suit = table_cards[0] // 9
            in_suit_indices = [i for i in range(36) if i // 9 == lead_suit]
            has_in_suit = any(hand[i] > 0.5 for i in in_suit_indices)
            
            if has_in_suit:
                for i in in_suit_indices:
                    if hand[i] > 0.5:
                        mask[i] = 1.0
            else:
                mask = hand.copy()
        
        return mask

    def clear_all(self):
        self.selected_hand_cards.clear()
        self.selected_table_cards = [None] * 4
        for btn in self.hand_buttons:
            btn.selected = False
            btn.update_appearance()
        for btn in self.table_buttons:
            btn.config(text=f"Player {btn.grid_info()['column']}\n(empty)", image="")
        self.update_hand_count()

    def complete_trick(self):
        table_idxs = [c for c in self.selected_table_cards if c is not None]
        if len(table_idxs) == 4:
            self.history.complete_trick(table_idxs, self.leader_var.get())
            self.clear_all()
            messagebox.showinfo("✅ Success", "Trick completed and added to history")

    def new_game(self):
        self.history.reset_full()
        self.clear_all()
        self.leader_var.set(0)
        self.player_var.set(0)
        self.trick_var.set(0)

    def update_trump(self, event=None):
        self.trump = MODES.index(self.trump_var.get())

class CardSelectionDialog(tk.Toplevel):
    def __init__(self, parent, title):
        super().__init__(parent)
        self.title(title)
        self.geometry("800x600")
        self.configure(bg='#2C3E50')
        self.result = None
        
        # Title
        tk.Label(self, text="🎴 Select a Card", font=('Arial', 18, 'bold'), 
                bg='#2C3E50', fg='#ECF0F1').pack(pady=20)
        
        # Create scrollable canvas for card grid
        canvas_frame = tk.Frame(self, bg='#2C3E50')
        canvas_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        canvas = tk.Canvas(canvas_frame, bg='#34495E', highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg='#34495E')
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Create card grid
        for i in range(36):
            if parent.card_images[i]:
                btn = tk.Button(scrollable_frame, image=parent.card_images[i],
                              command=lambda idx=i: self.select_card(idx),
                              relief='raised', bd=2, bg='#ECF0F1')
            else:
                btn = tk.Button(scrollable_frame, text=f"Card {i}", width=10, height=4,
                              command=lambda idx=i: self.select_card(idx),
                              font=('Arial', 10, 'bold'), bg='#ECF0F1', fg='#2C3E50')
            btn.grid(row=i//6, column=i%6, padx=3, pady=3)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Cancel button
        cancel_btn = tk.Button(self, text="❌ Cancel", command=self.cancel,
                              font=('Arial', 12, 'bold'), bg='#E74C3C', fg='white',
                              relief='raised', bd=2, padx=30, pady=10)
        cancel_btn.pack(pady=20)
        
        self.transient(parent)
        self.grab_set()
        parent.wait_window(self)

    def select_card(self, card_idx):
        self.result = card_idx
        self.destroy()

    def cancel(self):
        self.destroy()

if __name__ == "__main__":
    app = JassTestApp()
    app.mainloop()