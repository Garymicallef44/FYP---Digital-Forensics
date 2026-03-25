import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import threading
import itertools
import textwrap
import networkx as nx
import PatternMatching as pm
from ProvenanceAnalysis import ProvenanceAnalyzer


class ProvenanceGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Provenance Analysis")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 650)

        self.folderpath = tk.StringVar()
        self.comparison_results = []  # list of (pathA, pathB, evidenceAB, evidenceBA)
        self.analyzer = ProvenanceAnalyzer(similaritythreshold=25.0)
        self.photo_refs = [] 

        self._build_ui()


    def _build_ui(self):
        # Top toolbar
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill=tk.X)

        ttk.Label(toolbar, text="Image Folder:").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.folderpath, width=60).pack(side=tk.LEFT, padx=(4, 4))
        ttk.Button(toolbar, text="Browse…", command=self._browse_folder).pack(side=tk.LEFT)
        self.run_btn = ttk.Button(toolbar, text="Run Analysis", command=self._start_analysis)
        self.run_btn.pack(side=tk.LEFT, padx=(12, 0))

        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill=tk.X, padx=8)

        self.status_var = tk.StringVar(value="Select a folder and click Run Analysis.")
        ttk.Label(self.root, textvariable=self.status_var, anchor=tk.W).pack(fill=tk.X, padx=10)

        # Main paned area
        pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        pane.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        # left pair list
        left_frame = ttk.LabelFrame(pane, text="Comparison Pairs", padding=4)
        pane.add(left_frame, weight=1)

        self.pair_list = tk.Listbox(left_frame, activestyle="none", font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(left_frame, command=self.pair_list.yview)
        self.pair_list.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.pair_list.pack(fill=tk.BOTH, expand=True)
        self.pair_list.bind("<<ListboxSelect>>", self._on_pair_select)

        # right detail panel
        right_frame = ttk.LabelFrame(pane, text="Comparison Details", padding=8)
        pane.add(right_frame, weight=3)

        # image thumbnails row
        images_frame = ttk.Frame(right_frame)
        images_frame.pack(fill=tk.X, pady=(0, 8))

        left_img_frame = ttk.LabelFrame(images_frame, text="Image A", padding=2)
        left_img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 2))
        self.img_label_a = ttk.Label(left_img_frame, anchor=tk.CENTER)
        self.img_label_a.pack(fill=tk.BOTH, expand=True)
        self.name_label_a = ttk.Label(left_img_frame, anchor=tk.CENTER, wraplength=300)
        self.name_label_a.pack()

        right_img_frame = ttk.LabelFrame(images_frame, text="Image B", padding=2)
        right_img_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(2, 0))
        self.img_label_b = ttk.Label(right_img_frame, anchor=tk.CENTER)
        self.img_label_b.pack(fill=tk.BOTH, expand=True)
        self.name_label_b = ttk.Label(right_img_frame, anchor=tk.CENTER, wraplength=300)
        self.name_label_b.pack()

        # stats
        stats_frame = ttk.LabelFrame(right_frame, text="Match Statistics", padding=8)
        stats_frame.pack(fill=tk.BOTH, expand=True)

        self.stats_text = tk.Text(stats_frame, wrap=tk.WORD, height=14, state=tk.DISABLED,
                                  font=("Consolas", 10), bg="#f8f8f8", relief=tk.FLAT)
        self.stats_text.pack(fill=tk.BOTH, expand=True)

        # graph view
        graph_frame = ttk.LabelFrame(self.root, text="Graph View", padding=4)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
        self.graph_canvas = tk.Canvas(graph_frame, height=260, bg="white", highlightthickness=1,
                          highlightbackground="#d9d9d9")
        self.graph_canvas.pack(fill=tk.BOTH, expand=True)
        self.graph_canvas.bind("<Configure>", self._on_graph_resize)


    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if folder:
            self.folderpath.set(folder)
            
    def _start_analysis(self):
        folder = self.folderpath.get().strip()
        if not folder or not os.path.isdir(folder):
            messagebox.showwarning("Invalid Folder", "Please select a valid image folder.")
            return

        self.run_btn.configure(state=tk.DISABLED)
        self.pair_list.delete(0, tk.END)
        self.comparison_results.clear()
        self._clear_detail()
        self.status_var.set("Loading images…")
        self.progress["value"] = 0

        thread = threading.Thread(target=self._run_analysis, args=(folder,), daemon=True)
        thread.start()

    def _run_analysis(self, folder):
        try:
            imagepaths = sorted([
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            ])

            if len(imagepaths) < 2:
                self.root.after(0, lambda: messagebox.showwarning(
                    "Not Enough Images", "Need at least 2 images to compare."))
                self.root.after(0, lambda: self.run_btn.configure(state=tk.NORMAL))
                return

            # Load features
            featurecache = {}
            for i, path in enumerate(imagepaths):
                name = os.path.basename(path)
                self.root.after(0, lambda n=name: self.status_var.set(f"Extracting features: {n}"))
                img, pixels = pm.loadImage(path)
                kp, desc = pm.extractSIFT(img)
                featurecache[path] = (img, kp, desc, pixels)

            pairs = list(itertools.combinations(imagepaths, 2))
            total = len(pairs)
            results = []

            for idx, (p1, p2) in enumerate(pairs):
                n1, n2 = os.path.basename(p1), os.path.basename(p2)
                self.root.after(0, lambda a=n1, b=n2, i=idx, t=total:
                                self.status_var.set(f"Comparing ({i+1}/{t}): {a}  →  {b}"))

                ev12, ev21 = self.analyzer.comparePair(p1, p2, featurecache)

                results.append((p1, p2, ev12, ev21))
                pct = int((idx + 1) / total * 100)
                self.root.after(0, lambda v=pct: self.progress.configure(value=v))

            # Build provenance graph
            self.root.after(0, lambda: self.status_var.set("Building provenance graph…"))
            similarities = {(p1, p2): (ev12, ev21) for p1, p2, ev12, ev21 in results}
            self.analyzer.buildProvenanceGraph(similarities)
            self.analyzer.optimiseGraph()

            self.root.after(0, lambda: self._analysis_done(results))

        except Exception as exc:
            self.root.after(0, lambda e=str(exc): messagebox.showerror("Error", e))
            self.root.after(0, lambda: self.run_btn.configure(state=tk.NORMAL))

    def _analysis_done(self, results):
        self.comparison_results = results
        self.progress["value"] = 100
        self.run_btn.configure(state=tk.NORMAL)

        # Populate pair list
        for p1, p2, ev12, ev21 in results:
            n1 = os.path.basename(p1)
            n2 = os.path.basename(p2)
            best = max(ev12.score, ev21.score)
            self.pair_list.insert(tk.END, f"{n1}  →  {n2}   [{best:.1f}%]")

        self._draw_graph()
        self.status_var.set(f"Done – {len(results)} comparisons, "
                            f"{graph.number_of_edges()} graph edges.")

        if results:
            self.pair_list.selection_set(0)
            self._show_pair(0)

    def _on_pair_select(self, event):
        sel = self.pair_list.curselection()
        if sel:
            self._show_pair(sel[0])

    def _show_pair(self, index):
        p1, p2, ev12, ev21 = self.comparison_results[index]
        self.photo_refs.clear()

        # load thumbnails
        self._set_thumbnail(p1, self.img_label_a)
        self._set_thumbnail(p2, self.img_label_b)
        self.name_label_a.configure(text=os.path.basename(p1))
        self.name_label_b.configure(text=os.path.basename(p2))

        # stats text
        n1, n2 = os.path.basename(p1), os.path.basename(p2)
        best = max(ev12.score, ev21.score)
        direction_src, direction_dst = self.analyzer.inferDirection(p1, p2, ev12, ev21)

        lines = []
        lines.append(f"{'═' * 52}")
        lines.append(f"  Best Similarity Score:  {best:.2f}%")
        lines.append(f"{'═' * 52}")
        lines.append("")

        for label, nameA, nameB, ev in [("A → B", n1, n2, ev12), ("B → A", n2, n1, ev21)]:
            lines += self._format_direction_stats(label, nameA, nameB, ev)
        lines.append("")

        lines.append(f"  Inferred direction (source → derived):")
        lines.append(f"    {os.path.basename(direction_src)}  →  {os.path.basename(direction_dst)}")

        self._set_stats("\n".join(lines))

    def _format_direction_stats(self, label, nameA, nameB, ev):
        lines = []
        lines.append(f"  Direction  {label}  ({nameA} → {nameB})")
        lines.append(f"  ────────────────────────────────────────")
        lines.append(f"  Score:            {ev.score:.2f}%")
        lines.append(f"  Good matches:     {ev.good_matches}")
        lines.append(f"  Inliers (RANSAC): {ev.inliers}")
        if ev.good_matches:
            lines.append(f"  Inlier ratio:     {ev.inliers / ev.good_matches:.2%}")
        lines.append(f"  Keypoints {nameA}:      {ev.kp1}")
        lines.append(f"  Keypoints {nameB}:      {ev.kp2}")
        lines.append(f"  Homography det:   {ev.homography_det:.6f}")
        lines.append(f"  Pixels {nameA}:         {ev.img1pixels:,}")
        lines.append(f"  Pixels {nameB}:         {ev.img2pixels:,}")
        lines.append("")
        return lines

    def _set_thumbnail(self, path, label, max_size=400):
        img = cv2.imread(path)
        if img is None:
            label.configure(image="", text="(could not load)")
            return
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img.thumbnail((max_size, max_size), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        self.photo_refs.append(photo)
        label.configure(image=photo, text="")

    def _set_stats(self, text):
        self.stats_text.configure(state=tk.NORMAL)
        self.stats_text.delete("1.0", tk.END)
        self.stats_text.insert(tk.END, text)
        self.stats_text.configure(state=tk.DISABLED)

    def _on_graph_resize(self, _event):
        self._draw_graph()

    def _draw_graph(self):
        if not hasattr(self, "graph_canvas"):
            return

        graph = self.analyzer.graph
        canvas = self.graph_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 300)
        height = max(canvas.winfo_height(), 180)

        if graph.number_of_nodes() == 0:
            canvas.create_text(width // 2, height // 2,
                               text="No provenance graph available yet.",
                               fill="#555555", font=("Consolas", 11))
            return

        roots = set(self.analyzer.rootCandidates())
        pos = nx.spring_layout(graph, seed=42)

        node_radius = 18
        pad = 70  # Increased padding to accommodate labels and prevent cutoff

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        def scale_x(x):
            if max_x == min_x:
                return width / 2
            return pad + (x - min_x) / (max_x - min_x) * (width - 2 * pad)

        def scale_y(y):
            if max_y == min_y:
                return height / 2
            return pad + (y - min_y) / (max_y - min_y) * (height - 2 * pad)

        screen_pos = {n: (scale_x(x), scale_y(y)) for n, (x, y) in pos.items()}

        for u, v, data in graph.edges(data=True):
            x1, y1 = screen_pos[u]
            x2, y2 = screen_pos[v]

            dx, dy = x2 - x1, y2 - y1
            dist = max((dx * dx + dy * dy) ** 0.5, 1e-6)
            ux, uy = dx / dist, dy / dist

            sx = x1 + ux * node_radius
            sy = y1 + uy * node_radius
            ex = x2 - ux * node_radius
            ey = y2 - uy * node_radius

            canvas.create_line(sx, sy, ex, ey, arrow=tk.LAST, arrowshape=(14, 16, 6),
                               width=2.5, fill="#1f6fb2")

            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            label_id = canvas.create_text(mid_x, mid_y - 12, text=f"{data['weight']:.1f}%",
                                          fill="#0e3f6b", font=("Consolas", 9, "bold"))
            bx1, by1, bx2, by2 = canvas.bbox(label_id)
            canvas.create_rectangle(bx1 - 2, by1 - 1, bx2 + 2, by2 + 1,
                                    fill="white", outline="")
            canvas.tag_raise(label_id)

        for node, (x, y) in screen_pos.items():
            is_root = node in roots
            fill = "#d6f5d6" if is_root else "#f6fafe"
            outline = "#2b7a2b" if is_root else "#2a6fbb"
            width_px = 3 if is_root else 2
            node_name = os.path.basename(node)
            wrapped_name = "\n".join(textwrap.wrap(node_name, width=16)[:2])
            if len(textwrap.wrap(node_name, width=16)) > 2:
                wrapped_name += "..."

            canvas.create_oval(x - node_radius, y - node_radius,
                               x + node_radius, y + node_radius,
                               fill=fill, outline=outline, width=width_px)

            name_id = canvas.create_text(x, y + node_radius + 14, text=wrapped_name,
                                         fill="#0d1d2f", font=("Consolas", 9, "bold"),
                                         width=120, justify=tk.CENTER)
            nx1, ny1, nx2, ny2 = canvas.bbox(name_id)
            canvas.create_rectangle(nx1 - 4, ny1 - 2, nx2 + 4, ny2 + 2,
                                    fill="#ffffff", outline="#d9e1ea")
            canvas.tag_raise(name_id)

            if is_root:
                canvas.create_text(x, y - node_radius - 10, text="ROOT",
                                   fill="#2b7a2b", font=("Consolas", 8, "bold"))

    def _clear_detail(self):
        self.photo_refs.clear()
        self.img_label_a.configure(image="", text="")
        self.img_label_b.configure(image="", text="")
        self.name_label_a.configure(text="")
        self.name_label_b.configure(text="")
        self._set_stats("")
        if hasattr(self, "graph_canvas"):
            self.graph_canvas.delete("all")


if __name__ == "__main__":
    root = tk.Tk()
    app = ProvenanceGUI(root)
    root.mainloop()