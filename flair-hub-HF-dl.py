import os
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from huggingface_hub import HfApi, hf_hub_download

# ============================================================
# CONFIG
# ============================================================

DATASET_ID = "IGNF/FLAIR-HUB"
DEFAULT_DOWNLOAD_DIR = "./FLAIR-HUB_download"

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

api = HfApi()

checkbox_state = {}     
file_sizes = {}        
stop_flag = threading.Event()
dataset_files = []      


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def human_bytes(num):
    if num is None:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PB"


def parse_zip_metadata(path: str):
    """
    Parse data ZIP filename of the form:
        DOMAIN-YEAR_MODALITY_DATATYPE.zip

    Example:
        data/D004-2021_AERIAL_RGBI_IMS.zip
        -> domain=D004, year=2021, modality=AERIAL_RGBI, datatype=IMS
    """
    name = os.path.basename(path)
    if not name.lower().endswith(".zip"):
        return None

    if name == "GLOBAL_ALL_MTD.zip":
        return None

    stem = name[:-4]
    if "-" not in stem:
        return None

    domain, rest = stem.split("-", 1)
    parts = rest.split("_")
    year = parts[0] if parts and parts[0].isdigit() else None

    modality = None
    datatype = None

    if len(parts) >= 2:
        if len(parts) >= 3:
            datatype = parts[-1]
            modality = "_".join(parts[1:-1])
        else:
            modality = parts[1]

    return {
        "path": path,
        "name": name,
        "domain": domain,
        "year": year,
        "modality": modality,
        "datatype": datatype,
    }


# ============================================================
# MAIN WINDOW & STYLES
# ============================================================

root = tk.Tk()
root.title("FLAIR-HUB Dataset Download Tool")
root.geometry("1100x780")
root.minsize(860, 620)

BG_MAIN   = "#020617"  
BG_CARD   = "#111827"   
ACCENT    = "#98c220"   
ACCENT_H  = "#b4de34"  
FG_TEXT   = "#e5e7eb"  
FG_MUTED  = "#9ca3af" 
BORDER    = "#1f2937"  

LIST_BG   = "#f9fafb"
LIST_FG   = "#111827"
LIST_SEL_BG = "#d1fae5"   
LIST_SEL_FG = "#065f46"

style = ttk.Style()
if "clam" in style.theme_names():
    style.theme_use("clam")

BASE_FONT = ("Segoe UI", 10)
INFO_FONT = ("Consolas", 9)
LOG_FONT  = ("Consolas", 9)

root.option_add("*Font", BASE_FONT)
root.configure(bg=BG_MAIN)

style.configure("TFrame", background=BG_MAIN)
style.configure("Card.TFrame", background=BG_CARD)

style.configure(
    "Treeview",
    background=LIST_BG,
    fieldbackground=LIST_BG,
    foreground=LIST_FG,
    rowheight=24,
    borderwidth=0,
    highlightthickness=0,
)
style.map(
    "Treeview",
    background=[("selected", LIST_SEL_BG)],
    foreground=[("selected", LIST_SEL_FG)],
)

style.configure(
    "Accent.TButton",
    background="#111827",
    foreground=FG_TEXT,
    padding=(10, 4),
    relief="flat",
    borderwidth=1,
)
style.map(
    "Accent.TButton",
    background=[
        ("active", "#1f2937"),
        ("pressed", "#111827"),
        ("disabled", "#111827"),
    ],
    foreground=[
        ("disabled", FG_MUTED),
    ],
)

style.configure(
    "Primary.TButton",
    background=ACCENT,
    foreground="#030712",
    padding=(12, 5),
    relief="flat",
    borderwidth=0,
    font=("Segoe UI", 10, "bold"),
)
style.map(
    "Primary.TButton",
    background=[
        ("active", ACCENT_H),
        ("pressed", ACCENT),
        ("disabled", "#4b5563"),
    ],
    foreground=[
        ("disabled", "#111827"),
    ],
)

style.configure(
    "Accent.Horizontal.TProgressbar",
    troughcolor=BG_MAIN,
    bordercolor=BG_MAIN,
    background=ACCENT,
    lightcolor=ACCENT,
    darkcolor=ACCENT,
)

root.update_idletasks()
w = root.winfo_width()
h = root.winfo_height()
x = (root.winfo_screenwidth() // 2) - (w // 2)
y = (root.winfo_screenheight() // 2) - (h // 2)
root.geometry(f"{w}x{h}+{x}+{y}")


# ============================================================
# LAYOUT
# ============================================================

main_paned = ttk.Panedwindow(root, orient=tk.VERTICAL)
main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=(8, 4))

info_outer = tk.Frame(
    main_paned,
    bg=BG_MAIN,
)
info_frame = tk.Frame(
    info_outer,
    bg=BG_CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
    bd=0,
)
info_frame.pack(fill=tk.BOTH, expand=True)
main_paned.add(info_outer, weight=1)

info_label = tk.Label(
    info_frame,
    text="Dataset Information",
    font=("Segoe UI", 11, "bold"),
    bg=BG_CARD,
    fg=FG_TEXT,
)
info_label.pack(anchor="w", pady=(8, 4), padx=8)

info_text = tk.Text(
    info_frame,
    height=7,
    wrap="word",
    font=INFO_FONT,
    bg=BG_MAIN,
    fg=FG_TEXT,
    bd=0,
    relief="flat",
)
info_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))


# ============================================================
# FILES AREA (filters + two-column panel)
# ============================================================

files_outer = tk.Frame(root, bg=BG_MAIN)
files_card = tk.Frame(
    files_outer,
    bg=BG_CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
    bd=0,
)
files_card.pack(fill=tk.BOTH, expand=True)
main_paned.add(files_outer, weight=3)

tree_frame = ttk.Frame(files_card, style="Card.TFrame")
tree_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

filter_bar = ttk.Frame(tree_frame, padding=(4, 4), style="Card.TFrame")
filter_bar.pack(side=tk.TOP, fill=tk.X)

filter_title = ttk.Label(
    filter_bar,
    text="Filters",
    font=("Segoe UI", 10, "bold"),
    background=BG_CARD,
    foreground=FG_TEXT,
)
filter_title.grid(row=0, column=0, padx=(0, 8), sticky="w")

domain_var = tk.StringVar(value="All")
year_var = tk.StringVar(value="All")
modality_var = tk.StringVar(value="All")
datatype_var = tk.StringVar(value="All")

domain_label = ttk.Label(filter_bar, text="Domain", background=BG_CARD, foreground=FG_MUTED)
year_label = ttk.Label(filter_bar, text="Year", background=BG_CARD, foreground=FG_MUTED)
modality_label = ttk.Label(filter_bar, text="Modality", background=BG_CARD, foreground=FG_MUTED)
datatype_label = ttk.Label(filter_bar, text="Data type", background=BG_CARD, foreground=FG_MUTED)

domain_cb = ttk.Combobox(filter_bar, textvariable=domain_var, state="readonly", width=12, values=["All"])
year_cb = ttk.Combobox(filter_bar, textvariable=year_var, state="readonly", width=8, values=["All"])
modality_cb = ttk.Combobox(filter_bar, textvariable=modality_var, state="readonly", width=14, values=["All"])
datatype_cb = ttk.Combobox(filter_bar, textvariable=datatype_var, state="readonly", width=10, values=["All"])

domain_label.grid(row=0, column=1, sticky="w", padx=(0, 2))
domain_cb.grid(row=0, column=2, padx=(2, 8))

year_label.grid(row=0, column=3, sticky="w", padx=(0, 2))
year_cb.grid(row=0, column=4, padx=(2, 8))

modality_label.grid(row=0, column=5, sticky="w", padx=(0, 2))
modality_cb.grid(row=0, column=6, padx=(2, 8))

datatype_label.grid(row=0, column=7, sticky="w", padx=(0, 2))
datatype_cb.grid(row=0, column=8, padx=(2, 8))


def reset_filters():
    domain_var.set("All")
    year_var.set("All")
    modality_var.set("All")
    datatype_var.set("All")
    apply_filters()


def show_metadata_only():
    domain_var.set("GLOBAL")
    year_var.set("All")
    modality_var.set("GLOBAL")
    datatype_var.set("MTD")
    apply_filters()


metadata_btn = ttk.Button(filter_bar, text="Metadata", style="Accent.TButton", command=show_metadata_only)
metadata_btn.grid(row=0, column=9, padx=(4, 0), sticky="w")

reset_btn = ttk.Button(filter_bar, text="Reset", style="Accent.TButton", command=reset_filters)
reset_btn.grid(row=0, column=10, padx=(4, 0), sticky="w")

for col in range(11):
    filter_bar.columnconfigure(col, weight=0)

files_split = ttk.Panedwindow(tree_frame, orient=tk.HORIZONTAL)
files_split.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(4, 0))

left_files_frame = ttk.Frame(files_split, padding=(0, 4), style="Card.TFrame")
files_split.add(left_files_frame, weight=3)

left_label = ttk.Label(
    left_files_frame,
    text="Available files",
    font=("Segoe UI", 10, "bold"),
    background=BG_CARD,
    foreground=FG_TEXT,
)
left_label.pack(anchor="w", padx=(2, 0))

left_container = ttk.Frame(left_files_frame, style="Card.TFrame")
left_container.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

left_scroll = ttk.Scrollbar(left_container, orient=tk.VERTICAL)
left_scroll.pack(side=tk.RIGHT, fill=tk.Y)

file_tree = ttk.Treeview(left_container, yscrollcommand=left_scroll.set, selectmode="none")
file_tree.pack(fill=tk.BOTH, expand=True)
left_scroll.config(command=file_tree.yview)

file_tree["columns"] = ()
file_tree.column("#0", anchor=tk.W, stretch=True)
file_tree["show"] = "tree"

right_selected_frame = ttk.Frame(files_split, padding=(8, 4), style="Card.TFrame")
files_split.add(right_selected_frame, weight=2)

right_title_bar = tk.Frame(right_selected_frame, bg=BG_CARD)
right_title_bar.pack(fill=tk.X, pady=(0, 2))

right_label = tk.Label(
    right_title_bar,
    text="Selected files",
    font=("Segoe UI", 10, "bold"),
    bg=BG_CARD,
    fg=FG_TEXT,
)
right_label.pack(side=tk.LEFT)

badge_label = tk.Label(
    right_title_bar,
    text="[ 0 file — 0 ]",
    font=("Segoe UI", 9, "bold"),
    bg=ACCENT,      
    fg="#111827",   
    padx=8,
    pady=2,
)
badge_label.pack(side=tk.RIGHT, padx=4)


selected_container = ttk.Frame(right_selected_frame, style="Card.TFrame")
selected_container.pack(fill=tk.BOTH, expand=True, pady=(2, 0))

selected_scroll = ttk.Scrollbar(selected_container, orient=tk.VERTICAL)
selected_scroll.pack(side=tk.RIGHT, fill=tk.Y)

selected_listbox = tk.Listbox(
    selected_container,
    height=10,
    font=LOG_FONT,
    bg=LIST_BG,
    fg=LIST_FG,
    bd=0,
    relief="flat",
    selectbackground=LIST_SEL_BG,
    selectforeground=LIST_SEL_FG,
)
selected_listbox.pack(fill=tk.BOTH, expand=True)
selected_listbox.config(yscrollcommand=selected_scroll.set)
selected_scroll.config(command=selected_listbox.yview)

def update_selected_panel():
    """Refresh the right-hand 'selected files' list and the badge."""
    selected_listbox.delete(0, tk.END)

    total_bytes = 0
    selected_items = [m for m in dataset_files if checkbox_state.get(m["path"], False)]

    for meta in sorted(selected_items, key=lambda m: m["name"]):
        selected_listbox.insert(tk.END, meta["name"])
        if meta.get("size_bytes") is not None:
            total_bytes += meta["size_bytes"]

    count = len(selected_items)
    size_str = human_bytes(total_bytes) if total_bytes > 0 else "0"

    badge_label.config(
        text=f"[ {count} file{'s' if count != 1 else ''} — {size_str} ]"
    )


# ============================================================
# BOTTOM CONTROLS / LOGS
# ============================================================

bottom_outer = tk.Frame(root, bg=BG_MAIN)
bottom_card = tk.Frame(
    bottom_outer,
    bg=BG_CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
    bd=0,
)
bottom_card.pack(fill=tk.X, padx=10, pady=(2, 0))
bottom_outer.pack(fill=tk.X)

bottom_frame = ttk.Frame(bottom_card, padding=(10, 8), style="Card.TFrame")
bottom_frame.pack(fill=tk.X)

path_label = ttk.Label(bottom_frame, text="Download to:", background=BG_CARD, foreground=FG_MUTED)
path_label.grid(row=0, column=0, sticky="w")

download_path = ttk.Entry(bottom_frame)
download_path.insert(0, DEFAULT_DOWNLOAD_DIR)
download_path.grid(row=0, column=1, sticky="ew", padx=(6, 6))


def choose_folder():
    folder = filedialog.askdirectory()
    if folder:
        download_path.delete(0, tk.END)
        download_path.insert(0, folder)


browse_btn = ttk.Button(bottom_frame, text="📂 Browse…", style="Accent.TButton", command=choose_folder)
browse_btn.grid(row=0, column=2, sticky="ew")

btn_frame = ttk.Frame(bottom_frame, style="Card.TFrame")
btn_frame.grid(row=0, column=3, sticky="e", padx=(10, 0))

select_all_btn = ttk.Button(btn_frame, text="Select All", style="Accent.TButton")
unselect_all_btn = ttk.Button(btn_frame, text="Unselect All", style="Accent.TButton")
download_btn = ttk.Button(btn_frame, text="⬇ Download", style="Primary.TButton")
stop_btn = ttk.Button(btn_frame, text="⛔ Stop", style="Accent.TButton", state="disabled")

select_all_btn.pack(side=tk.LEFT, padx=(0, 4))
unselect_all_btn.pack(side=tk.LEFT, padx=(0, 4))
download_btn.pack(side=tk.LEFT, padx=(0, 4))
stop_btn.pack(side=tk.LEFT)

bottom_frame.columnconfigure(1, weight=1)

status_outer = tk.Frame(root, bg=BG_MAIN)
status_card = tk.Frame(
    status_outer,
    bg=BG_CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
    bd=0,
)
status_card.pack(fill=tk.X, padx=10, pady=(4, 0))
status_outer.pack(fill=tk.X)

status_frame = ttk.Frame(status_card, padding=(10, 4), style="Card.TFrame")
status_frame.pack(fill=tk.X)

status_label = ttk.Label(status_frame, text="Ready.", background=BG_CARD, foreground=FG_MUTED)
status_label.pack(side=tk.LEFT)

progress = ttk.Progressbar(status_frame, mode="indeterminate", length=220, style="Accent.Horizontal.TProgressbar")
progress.pack(side=tk.RIGHT)

log_outer = tk.Frame(root, bg=BG_MAIN)
log_card = tk.Frame(
    log_outer,
    bg=BG_CARD,
    highlightbackground=BORDER,
    highlightthickness=1,
    bd=0,
)
log_card.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 8))
log_outer.pack(fill=tk.BOTH, expand=True)

log_frame = ttk.Frame(log_card, padding=4, style="Card.TFrame")
log_frame.pack(fill=tk.BOTH, expand=True)

log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL)
log_scroll.pack(side=tk.RIGHT, fill=tk.Y)

log_text = tk.Text(
    log_frame,
    height=10,
    wrap="none",
    font=LOG_FONT,
    bg=BG_MAIN,
    fg=FG_TEXT,
    bd=0,
    relief="flat",
)
log_text.pack(fill=tk.BOTH, expand=True)
log_text.config(state="disabled")
log_scroll.config(command=log_text.yview)
log_text.config(yscrollcommand=log_scroll.set)


def append_log(line: str):
    log_text.config(state="normal")
    log_text.insert("end", line + "\n")
    log_text.see("end")
    log_text.config(state="disabled")


def clear_log():
    log_text.config(state="normal")
    log_text.delete("1.0", "end")
    log_text.config(state="disabled")


# ============================================================
# TREE / CHECKBOX LOGIC
# ============================================================

def refresh_checkboxes():
    """Update tree item text based on checkbox_state."""
    for iid in file_tree.get_children(""):
        checked = checkbox_state.get(iid, False)
        meta = next((f for f in dataset_files if f["path"] == iid), None)
        if meta is None:
            continue
        prefix = "☑ " if checked else "☐ "
        file_tree.item(iid, text=f"{prefix}{meta['name']}")
    update_selected_panel()


def apply_filters(*_):
    """Rebuild the left tree based on current filter values."""
    file_tree.delete(*file_tree.get_children(""))

    d_filter = domain_var.get()
    y_filter = year_var.get()
    m_filter = modality_var.get()
    t_filter = datatype_var.get()

    for meta in dataset_files:
        if d_filter != "All" and meta["domain"] != d_filter:
            continue
        if y_filter != "All" and meta["year"] != y_filter:
            continue
        if m_filter != "All" and meta["modality"] != m_filter:
            continue
        if t_filter != "All" and meta["datatype"] != t_filter:
            continue

        path = meta["path"]
        if path not in checkbox_state:
            checkbox_state[path] = False

        checked = checkbox_state[path]
        prefix = "☑ " if checked else "☐ "
        file_tree.insert("", "end", iid=path, text=f"{prefix}{meta['name']}")

    update_selected_panel()


def toggle_all(state: bool):
    """Select/unselect only currently visible (filtered) files."""
    for iid in file_tree.get_children(""):
        checkbox_state[iid] = state
    refresh_checkboxes()
    status_label.config(text=f"{'Selected' if state else 'Unselected'} filtered files.")


def on_tree_click(event):
    region = file_tree.identify("region", event.x, event.y)
    column = file_tree.identify_column(event.x)
    row = file_tree.identify_row(event.y)

    if region == "tree" and column == "#0" and row:
        bbox = file_tree.bbox(row)
        if bbox and event.x > bbox[0] + 20:
            current = checkbox_state.get(row, False)
            checkbox_state[row] = not current
            refresh_checkboxes()


file_tree.bind("<Button-1>", on_tree_click)


# ============================================================
# DATASET LOAD
# ============================================================

def load_dataset():
    global file_sizes, dataset_files
    info_text.delete(1.0, tk.END)
    file_tree.delete(*file_tree.get_children(""))
    checkbox_state.clear()
    file_sizes = {}
    dataset_files = []

    status_label.config(text="Contacting Hugging Face API…")
    root.update_idletasks()

    try:
        info = api.dataset_info(DATASET_ID, files_metadata=True)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load: {e}")
        status_label.config(text="Failed to load dataset.")
        return

    created = info.created_at.strftime("%Y-%m-%d %H:%M") if info.created_at else "N/A"
    modified = info.lastModified.strftime("%Y-%m-%d %H:%M") if info.lastModified else "N/A"
    size = getattr(info, "usedStorage", 0) or 0
    size_str = f"{size/1e9:.2f} GB" if size > 1e9 else f"{size/1e6:.2f} MB"

    meta_info = f"""
ID        : {info.id}
Author    : {info.author or 'Unknown'}
Created   : {created}
Modified  : {modified}
Size      : {size_str}
Downloads : {info.downloads or 0}
Likes     : {info.likes or 0}
Tags      : {', '.join(info.tags or [])}
Tasks     : {', '.join((info.cardData or {}).get('task_categories', []))}
    """.rstrip()

    info_text.insert(tk.END, meta_info)

    for s in info.siblings:
        path = s.rfilename

        if path.endswith("GLOBAL_ALL_MTD.zip"):
            size_attr = getattr(s, "size", None)
            m = {
                "path": path,
                "name": os.path.basename(path),
                "domain": "GLOBAL",
                "year": None,
                "modality": "GLOBAL",
                "datatype": "MTD",
                "size_bytes": size_attr,
            }
            dataset_files.append(m)
            if size_attr is not None:
                file_sizes[path] = size_attr
            continue

        if not path.startswith("data/"):
            continue
        if not path.lower().endswith(".zip"):
            continue

        m = parse_zip_metadata(path)
        if m is None:
            continue

        size_attr = getattr(s, "size", None)
        m["size_bytes"] = size_attr
        dataset_files.append(m)

        if size_attr is not None:
            file_sizes[path] = size_attr

    domains = sorted({f["domain"] for f in dataset_files if f["domain"]})
    years = sorted({f["year"] for f in dataset_files if f["year"]})
    modalities = sorted({f["modality"] for f in dataset_files if f["modality"]})
    datatypes = sorted({f["datatype"] for f in dataset_files if f["datatype"]})

    domain_cb["values"] = ["All"] + domains
    year_cb["values"] = ["All"] + years
    modality_cb["values"] = ["All"] + modalities
    datatype_cb["values"] = ["All"] + datatypes

    domain_var.set("All")
    year_var.set("All")
    modality_var.set("All")
    datatype_var.set("All")

    apply_filters()

    status_label.config(text=f"Loaded {len(dataset_files)} ZIP files from FLAIR-HUB (tiles + metadata).")
    append_log("=== Dataset loaded: FLAIR-HUB (data/*.zip + GLOBAL_ALL_MTD.zip) ===")


# ============================================================
# DOWNLOAD LOGIC
# ============================================================

def stop_download():
    stop_flag.set()
    status_label.config(text="⏹ Stop requested. Finishing current file…")
    stop_btn.config(state="disabled")
    append_log("⛔ Stop requested by user.")


def download_selected():
    folder = download_path.get().strip()
    os.makedirs(folder, exist_ok=True)

    files_to_download = [
        iid for iid in file_tree.get_children("")
        if checkbox_state.get(iid, False)
    ]

    if not files_to_download:
        messagebox.showinfo("No Selection", "No files selected (in current filter).")
        return

    total = len(files_to_download)
    stop_flag.clear()

    download_btn.config(state="disabled")
    select_all_btn.config(state="disabled")
    unselect_all_btn.config(state="disabled")
    browse_btn.config(state="disabled")
    stop_btn.config(state="normal")

    progress["mode"] = "indeterminate"
    progress.start(15)
    status_label.config(text=f"🚀 Launching download: {total} file(s)…")
    append_log(f"=== Download session started: {total} file(s) ===")

    def worker():
        t_session = time.time()
        processed = 0

        for index, fid in enumerate(files_to_download, start=1):
            if stop_flag.is_set():
                append_log("⛔ Download stopped by user.")
                break

            processed += 1

            meta = next((f for f in dataset_files if f["path"] == fid), None)
            name = meta["name"] if meta else fid
            planned = meta["size_bytes"] if meta else None
            size_info = human_bytes(planned) if planned else "unknown"

            def set_label(f=name, i=index):
                status_label.config(text=f"({i}/{total}) Downloading: {f}")
            root.after(0, set_label)

            root.after(
                0,
                lambda f=name, i=index, s=size_info: append_log(
                    f"▶ [{i}/{total}] {f} :: size ~ {s}"
                ),
            )

            t0 = time.time()
            local_path = None
            error = None

            try:
                local_path = hf_hub_download(
                    repo_id=DATASET_ID,
                    repo_type="dataset",
                    filename=fid,
                    local_dir=folder,
                    force_download=True,
                )
            except Exception as e:
                error = e

            elapsed = max(time.time() - t0, 1e-3)

            size_bytes = planned
            if size_bytes is None and local_path and os.path.exists(local_path):
                try:
                    size_bytes = os.path.getsize(local_path)
                except OSError:
                    size_bytes = None

            if size_bytes and elapsed > 0:
                speed_mbps = (size_bytes / elapsed) / (1024 * 1024)
            else:
                speed_mbps = None

            if error is None:
                size_h = human_bytes(size_bytes)
                if speed_mbps is not None:
                    summary = f"   ✅ Avg speed: {speed_mbps:.1f} MB/s | {size_h} in {elapsed:.1f}s"
                else:
                    summary = f"   ✅ {size_h} in {elapsed:.1f}s"
            else:
                summary = f"   ❌ ERROR: {error}"

            def push_summary(line=summary):
                append_log(line)
                append_log("")
            root.after(0, push_summary)

        def finish():
            progress.stop()
            stop_btn.config(state="disabled")
            download_btn.config(state="normal")
            select_all_btn.config(state="normal")
            unselect_all_btn.config(state="normal")
            browse_btn.config(state="normal")

            session_time = time.time() - t_session
            append_log(f"=== Download session ended :: {session_time:.1f}s ===")

            if stop_flag.is_set():
                status_label.config(text="⛔ Download stopped.")
                messagebox.showinfo("Stopped", "Downloads were stopped.")
            else:
                status_label.config(text=f"✅ Completed. Processed {processed} file(s).")
                messagebox.showinfo(
                    "Download Complete",
                    f"Download finished.\nProcessed {processed} file(s) in {session_time:.1f}s.",
                )

        root.after(0, finish)

    threading.Thread(target=worker, daemon=True).start()


# ============================================================
# WIRING
# ============================================================

select_all_btn.config(command=lambda: toggle_all(True))
unselect_all_btn.config(command=lambda: toggle_all(False))
download_btn.config(command=download_selected)
stop_btn.config(command=stop_download)

domain_cb.bind("<<ComboboxSelected>>", apply_filters)
year_cb.bind("<<ComboboxSelected>>", apply_filters)
modality_cb.bind("<<ComboboxSelected>>", apply_filters)
datatype_cb.bind("<<ComboboxSelected>>", apply_filters)

load_dataset()

root.mainloop()
