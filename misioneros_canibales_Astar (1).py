"""Misioneros y caníbales — versión 1.3. Requiere Python 3.10 o posterior.
Resuelve el problema con A* y una heurística de cruces mínimos, mostrando el
árbol y la frontera paso a paso; los cruces de la barca tienen una vista aparte.
Usa únicamente la biblioteca estándar: Tkinter/ttk/Canvas (Tcl/Tk 8.6 o posterior)
para la interfaz, heapq para la prioridad, dataclasses para los estados y
statistics/perf_counter para medir el tiempo sin animación. El índice comparativo
usa CR=cruces y CT=CE=expansiones por indicación de la docente; CB=CT+CE y
Total=CR+CB con pesos 1. CE es la convención de la práctica, no memoria medida.
En la interfaz el tiempo real es la mediana de 5 ejecuciones sin registrar eventos.
Ejecutar: python misioneros_canibales_Astar.py; añadir --consola -m 3 -c 3
para resolver desde la terminal sin abrir una ventana.
"""

from __future__ import annotations

import argparse
import heapq
from dataclasses import dataclass, field
from itertools import count
from statistics import median
from time import perf_counter


ROUTE_WEIGHT = 1
TIME_WEIGHT = 1
SPACE_WEIGHT = 1


@dataclass(frozen=True)
class State:
    missionaries_left: int
    cannibals_left: int
    boat: str

    def __str__(self) -> str:
        return f"({self.missionaries_left}, {self.cannibals_left}, {self.boat})"


@dataclass(frozen=True)
class Action:
    missionaries: int
    cannibals: int
    direction: str

    def __str__(self) -> str:
        return f"{self.missionaries} M + {self.cannibals} C  {self.direction}"


@dataclass(frozen=True)
class Node:
    identifier: int
    state: State
    g: int
    h: int
    parent: int | None = None
    action: Action | None = None

    @property
    def f(self) -> int:
        return self.g + self.h


@dataclass(frozen=True)
class SearchEvent:
    kind: str
    node_id: int
    message: str
    frontier: tuple[int, ...]


@dataclass
class Metrics:
    discovered: int = 0
    expanded: int = 0
    successors: int = 0
    discarded: int = 0
    improvements: int = 0
    frontier_peak: int = 0
    stored_peak: int = 0
    elapsed_ms: float = 0.0


@dataclass
class SearchResult:
    missionaries: int
    cannibals: int
    capacity: int | None
    status: str
    message: str
    nodes: list[Node] = field(default_factory=list)
    path: list[Node] = field(default_factory=list)
    events: list[SearchEvent] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)


@dataclass(frozen=True)
class Costs:
    route: int | None
    time: int
    space: int

    @property
    def search(self) -> float:
        return TIME_WEIGHT * self.time + SPACE_WEIGHT * self.space

    @property
    def total(self) -> float | None:
        return None if self.route is None else ROUTE_WEIGHT * self.route + self.search


def calculate_costs(result: SearchResult) -> Costs:
    route = result.path[-1].g if result.status == "resuelto" and result.path else None
    return Costs(route, result.metrics.expanded, result.metrics.expanded)


def safe_bank(missionaries: int, cannibals: int) -> bool:
    return missionaries >= 0 and cannibals >= 0 and (
        missionaries == 0 or missionaries >= cannibals
    )


def safe_state(state: State, missionaries: int, cannibals: int) -> bool:
    ml, cl = state.missionaries_left, state.cannibals_left
    return (
        state.boat in ("I", "D")
        and 0 <= ml <= missionaries and 0 <= cl <= cannibals
        and safe_bank(ml, cl)
        and safe_bank(missionaries - ml, cannibals - cl)
    )


def boat_capacity(missionaries: int, cannibals: int) -> int | None:
    if missionaries == cannibals == 0:
        return 0
    if missionaries == 0:
        return 2
    if cannibals > missionaries:
        return None
    if missionaries > cannibals or cannibals <= 3:
        return 2
    return 3 if cannibals <= 5 else 4


def valid_loads(capacity: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (m, c)
        for m in range(capacity + 1)
        for c in range(capacity + 1 - m)
        if m + c >= 1 and safe_bank(m, c)
    )


def successors(state: State, missionaries: int, cannibals: int,
               loads: tuple[tuple[int, int], ...]):
    left = state.boat == "I"
    available_m = state.missionaries_left if left else missionaries - state.missionaries_left
    available_c = state.cannibals_left if left else cannibals - state.cannibals_left
    sign = -1 if left else 1
    for m, c in loads:
        if m > available_m or c > available_c:
            continue
        target = State(state.missionaries_left + sign * m,
                       state.cannibals_left + sign * c, "D" if left else "I")
        if safe_state(target, missionaries, cannibals):
            yield Action(m, c, "I → D" if left else "D → I"), target


def is_goal(state: State) -> bool:
    return state == State(0, 0, "D")


def heuristic(state: State, capacity: int) -> int:
    remaining = state.missionaries_left + state.cannibals_left
    if remaining == 0:
        return 0
    if capacity <= 0:
        raise ValueError("La capacidad debe ser positiva si quedan personas.")
    return (remaining + capacity - 1) // capacity


def solve(missionaries: int, cannibals: int, *, capture_trace: bool = False) -> SearchResult:
    if type(missionaries) is not int or type(cannibals) is not int:
        raise ValueError("Las cantidades deben ser números enteros.")
    if missionaries < 0 or cannibals < 0:
        raise ValueError("Las cantidades no pueden ser negativas.")
    started = perf_counter()
    capacity = boat_capacity(missionaries, cannibals)
    result = SearchResult(missionaries, cannibals, capacity, "sin_solucion", "")
    if capacity is None:
        result.status = "inseguro"
        result.message = "Estado inicial inseguro: hay más caníbales que misioneros."
        result.metrics.elapsed_ms = (perf_counter() - started) * 1000
        return result

    initial = State(missionaries, cannibals, "I")
    root = Node(0, initial, 0, heuristic(initial, capacity))
    result.nodes.append(root)
    result.metrics.discovered = 1
    if capacity == 0:
        result.status = "resuelto"
        result.message = "No hay personas que transportar: 0 viajes. La barca permanece a la izquierda."
        result.path = [root]
        result.metrics.elapsed_ms = (perf_counter() - started) * 1000
        return result

    serial = count()
    frontier = [(root.f, root.h, next(serial), root.identifier)]
    best_g = {initial: 0}
    pending = {initial: root.identifier}
    loads = valid_loads(capacity)
    result.metrics.frontier_peak = 1
    result.metrics.stored_peak = 1

    def record(kind: str, node: Node, message: str) -> None:
        if capture_trace:
            ordered = tuple(sorted(pending.values(), key=lambda i: (
                result.nodes[i].f, result.nodes[i].h, i
            )))
            result.events.append(SearchEvent(kind, node.identifier, message, ordered))

    record("INICIO", root, f"Estado inicial {initial}. Prioridad: f={root.f}.")
    while frontier:
        _, _, _, identifier = heapq.heappop(frontier)
        node = result.nodes[identifier]
        if pending.get(node.state) != identifier:
            continue
        del pending[node.state]


        if is_goal(node.state):
            path = []
            cursor: Node | None = node
            while cursor is not None:
                path.append(cursor)
                cursor = result.nodes[cursor.parent] if cursor.parent is not None else None
            result.path = path[::-1]
            result.status = "resuelto"
            result.message = f"Ruta óptima encontrada: {node.g} viajes."
            record("META", node, result.message)
            break

        result.metrics.expanded += 1
        record("EXPANDIR", node,
               f"Expandir {node.state}: g={node.g}, h={node.h}, f={node.f}.")
        for action, target in successors(node.state, missionaries, cannibals, loads):
            result.metrics.successors += 1
            new_g = node.g + 1
            previous_g = best_g.get(target)
            if previous_g is not None and new_g >= previous_g:
                result.metrics.discarded += 1
                record("DESCARTAR", node,
                       f"{action} → {target}: g={new_g}; ya se conoce g={previous_g}.")
                continue

            child = Node(len(result.nodes), target, new_g, heuristic(target, capacity),
                         node.identifier, action)
            result.nodes.append(child)
            result.metrics.stored_peak = max(result.metrics.stored_peak, len(result.nodes))
            best_g[target] = new_g
            pending[target] = child.identifier
            heapq.heappush(frontier, (child.f, child.h, next(serial), child.identifier))
            if previous_g is None:
                result.metrics.discovered += 1
            else:
                result.metrics.improvements += 1
            result.metrics.frontier_peak = max(result.metrics.frontier_peak, len(pending))
            record("GENERAR" if previous_g is None else "MEJORAR", child,
                   f"{action} → {target}: g={child.g}, h={child.h}, f={child.f}.")
    else:
        result.message = "Se agotó la frontera: no existe una ruta con esta capacidad."
    result.metrics.elapsed_ms = (perf_counter() - started) * 1000
    return result


class Application:
    BG = "#101214"
    PANEL = "#181b1e"
    FIELD = "#22272b"
    BORDER = "#343b40"
    TEXT = "#e3e8e7"
    MUTED = "#919c9e"
    GREEN = "#668b70"
    NEON = "#a3ff42"
    BLUE = "#5283a0"
    MAROON = "#a63d53"

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk, messagebox, font
        self.tk, self.ttk, self.messagebox = tk, ttk, messagebox
        self.root = root
        families = set(font.families(root))
        self.font_family = "Segoe UI" if "Segoe UI" in families else "DejaVu Sans"
        self.result: SearchResult | None = None
        self.event_index = -1
        self.route_index = 0
        self.timer = None
        self.playing = False
        self.visible_nodes: set[int] = set()
        self.expanded_nodes: set[int] = set()
        self.positions: dict[int, tuple[float, float]] = {}
        self.zoom = 1.0
        root.title("Misioneros y caníbales · A* · v1.3")
        root.geometry(f"{min(1280, root.winfo_screenwidth()-60)}x{min(900, root.winfo_screenheight()-80)}")
        root.minsize(980, 740)
        root.configure(bg=self.BG)
        self.configure_style()
        shell = ttk.Frame(root, padding=(22, 18))
        shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell)
        header.pack(fill="x", pady=(0, 16))
        ttk.Label(header, text="Misioneros y caníbales", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="BÚSQUEDA A*", style="Accent.TLabel").pack(side="right", padx=2)
        config = ttk.Frame(shell, style="Panel.TFrame", padding=(14, 8))
        config.pack(fill="x", pady=(0, 14))
        self.m_var, self.c_var = tk.StringVar(value="3"), tk.StringVar(value="3")
        for label, var in (("Misioneros", self.m_var), ("Caníbales", self.c_var)):
            ttk.Label(config, text=label, style="Panel.TLabel").pack(side="left", padx=(0, 8))
            ttk.Spinbox(config, from_=0, to=20, width=5, textvariable=var).pack(side="left", padx=(0, 22))
        ttk.Button(config, text="Resolver", style="Primary.TButton", command=self.run_search).pack(side="left")
        capacity_box = tk.Frame(config, bg="#253b2f", highlightbackground="#668b70", highlightthickness=1,
                                padx=18, pady=5)
        capacity_box.pack(side="right")
        tk.Label(capacity_box, text="CAPACIDAD DEL BOTE", bg="#253b2f", fg="#bbd5c3", bd=0, pady=0,
                 font=(self.font_family, 8, "bold")).pack()
        self.capacity_var = tk.StringVar(value="Automática")
        tk.Label(capacity_box, textvariable=self.capacity_var, bg="#253b2f", fg=self.TEXT, bd=0, pady=0,
                 font=(self.font_family, 17, "bold")).pack()
        self.build_cost_panel(shell)
        self.tabs = ttk.Notebook(shell)
        self.tabs.pack(fill="both", expand=True)
        self.search_tab = ttk.Frame(self.tabs, padding=(12, 12))
        self.route_tab = ttk.Frame(self.tabs, padding=(12, 12))
        self.tabs.add(self.search_tab, text="  Árbol de búsqueda  ")
        self.tabs.add(self.route_tab, text="  Cruces de la barca  ")
        self.build_search_tab()
        self.build_route_tab()
        self.tabs.bind("<<NotebookTabChanged>>", lambda _: self.pause())
        root.protocol("WM_DELETE_WINDOW", self.close)

    def build_cost_panel(self, parent):
        strip = self.ttk.Frame(parent)
        strip.pack(fill="x", pady=(0, 4))
        self.cost_values = {}
        self.cost_details = {}
        entries = (("route", "CR · COSTO DE RUTA", "Cada cruce cuesta 1"),
                   ("search", "CB · COSTO DE BÚSQUEDA", "CT = CE = expansiones"),
                   ("total", "COSTO TOTAL · ÍNDICE", "CR + CB"))
        for col, (key, title, detail) in enumerate(entries):
            strip.columnconfigure(col, weight=1, uniform="costs")
            box = self.tk.Frame(strip, bg=self.PANEL, highlightbackground=self.BORDER,
                                highlightthickness=1, padx=10, pady=4)
            box.grid(row=0, column=col, sticky="nsew", padx=(0 if col==0 else 5, 0 if col==2 else 5))
            self.tk.Label(box, text=title, bg=self.PANEL, fg=self.MUTED, bd=0, padx=0, pady=0,
                          font=(self.font_family, 8, "bold")).pack(anchor="w")
            value = self.tk.StringVar(value="—")
            description = self.tk.StringVar(value=detail)
            self.cost_values[key], self.cost_details[key] = value, description
            self.tk.Label(box, textvariable=value, bg=self.PANEL, fg="#b7d0bf", bd=0, padx=0, pady=0,
                          font=(self.font_family, 15, "bold")).pack(anchor="w", pady=(1, 0))
            self.tk.Label(box, textvariable=description, bg=self.PANEL, fg=self.MUTED, bd=0, padx=0, pady=0,
                          font=(self.font_family, 8)).pack(anchor="w")
        self.cost_note = self.tk.StringVar(value="Índice comparativo: CB = CT + CE; Total = CR + CB. Pesos 1; el tiempo en ms se muestra aparte.")
        self.ttk.Label(parent, textvariable=self.cost_note, style="Small.TLabel").pack(anchor="w", pady=(2, 10))

    def update_cost_panel(self):
        if self.result is None:
            return
        cost = calculate_costs(self.result)
        self.cost_values["route"].set(f"{cost.route} cruces" if cost.route is not None else "Sin ruta")
        self.cost_values["search"].set(f"{cost.search:g} puntos")
        self.cost_values["total"].set(f"{cost.total:g} puntos" if cost.total is not None else "No aplicable")
        self.cost_details["search"].set(f"CT: {cost.time} · CE: {cost.space} · ambos = expansiones")
        self.cost_details["total"].set(f"{ROUTE_WEIGHT:g}×CR + {TIME_WEIGHT:g}×CT + {SPACE_WEIGHT:g}×CE")
        self.cost_note.set(f"CB = {TIME_WEIGHT:g}×CT + {SPACE_WEIGHT:g}×CE. CT = CE según la práctica. "
                           f"Tiempo sin traza: {self.result.metrics.elapsed_ms:.3f} ms (mediana de 5).")

    def configure_style(self):
        style = self.ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", background=self.BG, foreground=self.TEXT, font=(self.font_family, 10))
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT)
        style.configure("Title.TLabel", font=(self.font_family, 23, "bold"))
        style.configure("Accent.TLabel", foreground=self.GREEN, font=(self.font_family, 11, "bold"))
        style.configure("Muted.TLabel", foreground=self.MUTED)
        style.configure("Panel.TLabel", background=self.PANEL)
        style.configure("PanelMuted.TLabel", background=self.PANEL, foreground=self.MUTED)
        style.configure("Small.TLabel", foreground=self.MUTED, font=(self.font_family, 9))
        style.configure("TButton", background=self.FIELD, foreground=self.TEXT, padding=(14, 8),
                        bordercolor=self.BORDER, lightcolor=self.FIELD, darkcolor=self.FIELD, borderwidth=1)
        style.map("TButton", background=[("active", "#30383d"), ("disabled", self.PANEL)],
                  foreground=[("disabled", "#566064")])
        style.configure("Primary.TButton", background="#294333", bordercolor="#466a50",
                        lightcolor="#294333", darkcolor="#294333", font=(self.font_family, 10, "bold"))
        style.map("Primary.TButton", background=[("active", "#365840"), ("disabled", self.PANEL)])
        style.configure("TSpinbox", fieldbackground=self.FIELD, foreground=self.TEXT,
                        arrowcolor=self.MUTED, background=self.FIELD, bordercolor=self.BORDER,
                        lightcolor=self.FIELD, darkcolor=self.FIELD, padding=5)
        style.map("TSpinbox", fieldbackground=[("readonly", self.FIELD)],
                  background=[("active", self.BORDER)], arrowcolor=[("active", self.TEXT)])
        style.configure("TNotebook", background=self.BG, borderwidth=0, bordercolor=self.BORDER,
                        lightcolor=self.BORDER, darkcolor=self.BORDER, tabmargins=(0, 0, 0, 0))
        style.configure("TNotebook.Tab", background=self.PANEL, foreground=self.MUTED, padding=(12, 9),
                        bordercolor=self.BORDER, lightcolor=self.BORDER)
        style.map("TNotebook.Tab", background=[("selected", self.FIELD), ("active", "#252d30")],
                  foreground=[("selected", self.TEXT)])
        style.configure("Treeview", background=self.PANEL, fieldbackground=self.PANEL,
                        foreground=self.TEXT, rowheight=32, borderwidth=0, bordercolor=self.PANEL)
        style.configure("Treeview.Heading", background=self.FIELD, foreground=self.MUTED,
                        font=(self.font_family, 9, "bold"), padding=8, bordercolor=self.BORDER,
                        lightcolor=self.FIELD, darkcolor=self.FIELD)
        style.map("Treeview", background=[("selected", "#203746")], foreground=[("selected", self.TEXT)])
        style.map("Treeview.Heading", background=[("active", self.FIELD)])
        for orientation in ("Vertical", "Horizontal"):
            style.configure(f"{orientation}.TScrollbar", background=self.BORDER,
                            troughcolor=self.PANEL, bordercolor=self.PANEL,
                            arrowcolor=self.MUTED, lightcolor=self.BORDER, darkcolor=self.BORDER)
            style.map(f"{orientation}.TScrollbar", background=[("active", "#4a555b"), ("pressed", "#607079")])
        style.configure("TPanedwindow", background=self.BORDER)

    def make_table(self, parent, columns, widths, height=8, expand=True):
        frame = self.ttk.Frame(parent, style="Panel.TFrame")
        frame.pack(fill="both" if expand else "x", expand=expand)
        tree = self.ttk.Treeview(frame, columns=columns, show="headings", height=height, selectmode="browse")
        for column, width in zip(columns, widths):
            tree.heading(column, text=column)
            tree.column(column, width=width, minwidth=30, anchor="center", stretch=(column == "Estado"))
        scroll = self.ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        tree.tag_configure("next", background="#203746", foreground="#c2dce9")
        return tree

    def build_search_tab(self):
        ttk = self.ttk
        formula = ttk.Frame(self.search_tab, style="Panel.TFrame", padding=(14, 11))
        formula.pack(fill="x", pady=(0, 12))
        ttk.Label(formula, text="f(x) = g(x) + h(x)     ·     h(x) = ⌈(Mi + Ci) / C⌉",
                  style="Panel.TLabel", font=(self.font_family, 12, "bold")).pack(anchor="w")
        ttk.Label(formula, text="g: cruces realizados · h: cruces restantes estimados · Mi/Ci: personas a la izquierda · C: capacidad",
                  style="PanelMuted.TLabel", font=(self.font_family, 9)).pack(anchor="w", pady=(5, 0))
        toolbar = ttk.Frame(self.search_tab)
        toolbar.pack(fill="x", pady=(0, 10))
        self.search_play = ttk.Button(toolbar, text="Reproducir", command=lambda: self.toggle_play("search"), state="disabled")
        self.search_play.pack(side="left", padx=(0, 7))
        self.step_button = ttk.Button(toolbar, text="Siguiente paso", command=self.manual_event, state="disabled")
        self.step_button.pack(side="left", padx=(0, 7))
        self.restart_button = ttk.Button(toolbar, text="Reiniciar", command=self.reset_search, state="disabled")
        self.restart_button.pack(side="left")
        self.progress_var = self.tk.StringVar(value="Listo para comenzar")
        ttk.Label(toolbar, textvariable=self.progress_var, style="Muted.TLabel").pack(side="right")
        self.panes = ttk.Panedwindow(self.search_tab, orient="horizontal")
        self.panes.pack(fill="both", expand=True)
        graph_panel = ttk.Frame(self.panes, style="Panel.TFrame", padding=10)
        frontier_panel = ttk.Frame(self.panes, style="Panel.TFrame", padding=10)
        self.panes.add(graph_panel, weight=4)
        self.panes.add(frontier_panel, weight=1)
        self.root.after_idle(self.position_sash)
        ttk.Label(graph_panel, text="ÁRBOL DE EXPLORACIÓN", style="PanelMuted.TLabel",
                  font=(self.font_family, 9, "bold")).pack(anchor="w", pady=(1, 9))
        graph_frame = ttk.Frame(graph_panel, style="Panel.TFrame")
        graph_frame.pack(fill="both", expand=True)
        graph_frame.rowconfigure(0, weight=1)
        graph_frame.columnconfigure(0, weight=1)
        self.graph = self.tk.Canvas(graph_frame, bg="#131719", height=150, highlightthickness=0)
        self.graph.grid(row=0, column=0, sticky="nsew")
        ys = ttk.Scrollbar(graph_frame, orient="vertical", command=self.graph.yview)
        xs = ttk.Scrollbar(graph_frame, orient="horizontal", command=self.graph.xview)
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        self.graph.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        self.graph.bind("<MouseWheel>", self.zoom_graph)
        self.graph.bind("<Button-4>", lambda e: self.zoom_graph(e, 1.1))
        self.graph.bind("<Button-5>", lambda e: self.zoom_graph(e, 1/1.1))
        self.graph.bind("<ButtonPress-1>", lambda e: self.graph.scan_mark(e.x, e.y))
        self.graph.bind("<B1-Motion>", lambda e: self.graph.scan_dragto(e.x, e.y, gain=1))
        legend = ttk.Frame(graph_panel, style="Panel.TFrame")
        legend.pack(side="bottom", fill="x", pady=(7, 0), before=graph_frame)
        for label, color in (("● Actual", self.BLUE), ("● Pendiente", self.GREEN),
                              ("● Expandido", self.MUTED), ("● Solución", self.NEON)):
            ttk.Label(legend, text=label, foreground=color, style="Panel.TLabel",
                      font=(self.font_family, 9)).pack(side="left", padx=(0, 12))
        ttk.Label(graph_panel, text="Rueda: zoom · Arrastra para mover", style="PanelMuted.TLabel",
                  font=(self.font_family, 9)).pack(side="bottom", anchor="w", pady=(5, 0), before=legend)
        self.table_title = self.tk.StringVar(value="FRONTERA")
        self.table_hint = self.tk.StringVar(value="Primera fila: próximo estado\nOrden: f → h → llegada")
        ttk.Label(frontier_panel, textvariable=self.table_title, style="PanelMuted.TLabel",
                  font=(self.font_family, 9, "bold")).pack(anchor="w", pady=(1, 6))
        ttk.Label(frontier_panel, textvariable=self.table_hint, style="PanelMuted.TLabel",
                  font=(self.font_family, 9)).pack(anchor="w", pady=(0, 12))
        self.frontier_table = self.make_table(frontier_panel, ("Paso", "Estado", "g", "h", "f"),
                                               (40, 111, 34, 34, 34), height=3)
        self.frontier_table.configure(displaycolumns=("Estado", "g", "h", "f"))
        self.frontier_table.tag_configure("goal", background="#263d1c", foreground=self.NEON)
        self.frontier_table.bind("<Configure>", self.fit_search_table)
        self.event_var = self.tk.StringVar(value="Configura las cantidades y pulsa Resolver para construir el árbol.")
        self.event_label = ttk.Label(self.search_tab, textvariable=self.event_var, style="Muted.TLabel",
                                     wraplength=1140, padding=(0, 12, 0, 0))
        self.event_label.pack(side="bottom", fill="x", before=self.panes)
        self.search_tab.bind("<Configure>", lambda e: self.event_label.configure(wraplength=max(300, e.width-30)))
        self.draw_empty_graph()

    def position_sash(self):
        width = self.panes.winfo_width()
        if width > 600:
            self.panes.sashpos(0, width-310)

    def fit_search_table(self, event=None):
        available = self.frontier_table.winfo_width()-4
        if available < 180:
            return
        final = "Paso" in self.frontier_table["displaycolumns"]
        self.frontier_table.column("Paso", width=56, minwidth=56, stretch=False)
        for column in ("g", "h", "f"):
            self.frontier_table.column(column, width=35, minwidth=30, stretch=False)
        self.frontier_table.column("Estado", width=max(65, available-105-(56 if final else 0)),
                                    minwidth=65, stretch=False)

    def build_route_tab(self):
        ttk = self.ttk
        bar = ttk.Frame(self.route_tab)
        bar.pack(fill="x", pady=(0, 12))
        self.route_buttons = []
        for text, command in (("Anterior", self.previous_route), ("Siguiente cruce", self.manual_route),
                              ("Reproducir", lambda: self.toggle_play("route")), ("Reiniciar", self.reset_route)):
            button = ttk.Button(bar, text=text, command=command, state="disabled")
            button.pack(side="left", padx=(0, 7))
            self.route_buttons.append(button)
        self.route_play = self.route_buttons[2]
        self.route_label_var = self.tk.StringVar(value="La solución aparecerá después de resolver el problema.")
        ttk.Label(self.route_tab, textvariable=self.route_label_var, style="Muted.TLabel").pack(anchor="w", pady=(0, 12))
        self.river = self.tk.Canvas(self.route_tab, height=265, bg=self.PANEL, highlightthickness=0)
        self.river.pack(fill="x", pady=(0, 15))
        self.river.bind("<Configure>", lambda _: self.draw_river())
        self.route_table = self.make_table(self.route_tab,
            ("Paso", "Viaje", "Izquierda M / C", "Derecha M / C", "Barca"), (65, 220, 200, 200, 85),
            height=4, expand=False)
        for col in self.route_table["columns"]:
            self.route_table.column(col, stretch=True)
        self.route_table.bind("<<TreeviewSelect>>", self.select_route)

    @staticmethod
    def clear_table(table):
        children = table.get_children()
        if children:
            table.delete(*children)

    def run_search(self):
        self.pause()
        try:
            m, c = int(self.m_var.get()), int(self.c_var.get())
            if not (0 <= m <= 20 and 0 <= c <= 20):
                raise ValueError
        except ValueError:
            self.messagebox.showerror("Entrada inválida", "Ingresa enteros de 0 a 20 en ambos campos.")
            return
        self.result = solve(m, c, capture_trace=True)
        self.result.metrics.elapsed_ms = median(solve(m, c).metrics.elapsed_ms for _ in range(5))
        capacity = self.result.capacity
        self.capacity_var.set(f"{capacity} plazas" if capacity is not None else "No aplicable")
        self.update_cost_panel()
        self.calculate_positions()
        self.clear_table(self.route_table)
        for step, node in enumerate(self.result.path):
            state = node.state
            self.route_table.insert("", "end", iid=str(step), values=(
                step, str(node.action) if node.action else "Inicio",
                f"{state.missionaries_left} / {state.cannibals_left}",
                f"{m-state.missionaries_left} / {c-state.cannibals_left}", state.boat))
        has_events = bool(self.result.events)
        for button in (self.search_play, self.step_button, self.restart_button):
            button.configure(state="normal" if has_events else "disabled")
        for button in self.route_buttons:
            button.configure(state="normal" if len(self.result.path)>1 else "disabled")
        self.reset_route()
        self.reset_search()

    def calculate_positions(self):
        self.positions.clear()
        if not self.result or not self.result.nodes:
            return
        children = {node.identifier: [] for node in self.result.nodes}
        for node in self.result.nodes:
            if node.parent is not None:
                children[node.parent].append(node.identifier)
        widths = {}
        for node in reversed(self.result.nodes):
            group = children[node.identifier]
            widths[node.identifier] = max(210, sum(widths[i] for i in group)+32*max(0, len(group)-1))
        stack = [(0, 30.0)]
        while stack:
            identifier, left = stack.pop()
            node = self.result.nodes[identifier]
            self.positions[identifier] = (left+(widths[identifier]-210)/2, 30+node.g*185)
            cursor = left
            for child in children[identifier]:
                stack.append((child, cursor))
                cursor += widths[child]+32

    def reset_search(self):
        self.pause()
        self.event_index = -1
        self.visible_nodes.clear()
        self.expanded_nodes.clear()
        self.zoom = 1.0
        self.clear_table(self.frontier_table)
        self.table_title.set("FRONTERA")
        self.table_hint.set("Primera fila: próximo estado\nOrden: f → h → llegada")
        self.frontier_table.configure(displaycolumns=("Estado", "g", "h", "f"))
        self.fit_search_table()
        self.event_label.configure(foreground=self.MUTED)
        if self.result and self.result.events:
            self.search_play.configure(state="normal")
            self.step_button.configure(state="normal")
            self.next_event()
        else:
            self.draw_empty_graph()
            self.progress_var.set("Sin búsqueda")
            self.event_var.set(self.result.message if self.result else "Pulsa Resolver para comenzar.")
            if self.result and self.result.path:
                self.show_final_path_table()
            elif self.result:
                self.table_title.set("SIN RUTA")
                self.table_hint.set("No se inició la búsqueda.")

    def draw_empty_graph(self):
        self.graph.delete("all")
        self.graph.configure(scrollregion=(0, 0, 700, 300))
        self.graph.create_text(30, 36, anchor="nw", text="El árbol se construirá aquí, paso a paso.",
                               fill=self.MUTED, font=(self.font_family, 12))

    def next_event(self):
        if not self.result or self.event_index+1 >= len(self.result.events):
            return False
        self.event_index += 1
        event = self.result.events[self.event_index]
        if event.kind in ("INICIO", "GENERAR", "MEJORAR"):
            self.visible_nodes.add(event.node_id)
        if event.kind == "EXPANDIR":
            self.expanded_nodes.add(event.node_id)
        self.draw_search()
        if self.event_index == len(self.result.events)-1:
            self.pause()
            self.search_play.configure(state="disabled")
            self.step_button.configure(state="disabled")
        return True

    def manual_event(self):
        self.pause()
        self.next_event()

    def show_final_path_table(self):
        self.clear_table(self.frontier_table)
        self.table_title.set("RUTA SOLUCIÓN")
        self.table_hint.set("Del estado inicial a la meta\nValores de cada nodo del camino")
        self.frontier_table.configure(displaycolumns=("Paso", "Estado", "g", "h", "f"))
        self.fit_search_table()
        if not self.result:
            return
        for step, node in enumerate(self.result.path):
            self.frontier_table.insert("", "end", iid=f"path-{step}",
                                       values=(step, str(node.state), node.g, node.h, node.f),
                                       tags=("goal",) if step==len(self.result.path)-1 else ())

    def draw_search(self, focus=True):
        if not self.result or self.event_index < 0:
            return
        r = self.result
        event = r.events[self.event_index]
        self.progress_var.set(f"Paso {self.event_index+1} / {len(r.events)}")
        self.event_var.set(event.message)
        self.event_label.configure(foreground=self.NEON if event.kind=="META" else self.MUTED)
        self.clear_table(self.frontier_table)
        if event.kind == "META":
            self.show_final_path_table()
        else:
            for index, identifier in enumerate(event.frontier):
                node = r.nodes[identifier]
                self.frontier_table.insert("", "end", values=("", str(node.state), node.g, node.h, node.f),
                                           tags=("next",) if index==0 else ())
        self.graph.delete("all")
        path_ids = {n.identifier for n in r.path} if event.kind == "META" else set()
        z = self.zoom
        body_font = (self.font_family, max(5, round(10*z)))
        small_font = (self.font_family, max(5, round(9*z)))
        for identifier in sorted(self.visible_nodes):
            node = r.nodes[identifier]
            x, y = self.positions[identifier]
            if node.parent is not None and node.parent in self.visible_nodes:
                px, py = self.positions[node.parent]
                solution = identifier in path_ids and node.parent in path_ids
                color = self.NEON if solution else "#425056"
                self.graph.create_line((px+105)*z, (py+112)*z, (x+105)*z, y*z,
                                       fill=color, width=max(1, (3 if solution else 1.5)*z), arrow="last")
                tx, ty = (px+x)/2+105, (py+112+y)/2
                label = self.graph.create_text(tx*z, ty*z, text=str(node.action),
                                                fill=self.NEON if solution else self.MUTED, font=small_font)
                bounds = self.graph.bbox(label)
                if bounds:
                    background = self.graph.create_rectangle(bounds[0]-4, bounds[1]-2, bounds[2]+4,
                                                              bounds[3]+2, fill="#131719", outline="")
                    self.graph.tag_lower(background, label)
        for identifier in sorted(self.visible_nodes):
            node = r.nodes[identifier]
            x, y = self.positions[identifier]
            fill, outline = "#202823", self.GREEN
            if identifier in self.expanded_nodes:
                fill, outline = "#22272b", "#606b70"
            if identifier == event.node_id:
                fill, outline = "#1c3040", self.BLUE
            if identifier in path_ids:
                fill, outline = "#20321d", self.NEON
            self.graph.create_rectangle(x*z, y*z, (x+210)*z, (y+112)*z,
                                        fill=fill, outline=outline, width=max(1, 2*z))
            self.graph.create_text((x+105)*z, (y+23)*z, text=str(node.state),
                                   fill=self.NEON if identifier in path_ids else self.TEXT,
                                   font=(self.font_family, max(6, round(12*z)), "bold"))
            self.graph.create_line((x+12)*z, (y+43)*z, (x+198)*z, (y+43)*z, fill=self.BORDER)
            self.graph.create_text((x+105)*z, (y+63)*z, text=f"g = {node.g}    h = {node.h}    f = {node.f}",
                                   fill=self.TEXT, font=body_font)
            state = node.state
            banks = (f"I: {state.missionaries_left}M · {state.cannibals_left}C     "
                     f"D: {r.missionaries-state.missionaries_left}M · {r.cannibals-state.cannibals_left}C")
            self.graph.create_text((x+105)*z, (y+91)*z, text=banks, fill=self.MUTED, font=small_font)
        bounds = self.graph.bbox("all")
        if bounds:
            region = (bounds[0]-30*z, bounds[1]-30*z, bounds[2]+30*z, bounds[3]+30*z)
            if region[2]-region[0] < self.graph.winfo_width():
                padding = (self.graph.winfo_width()-(region[2]-region[0]))/2
                region = (region[0]-padding, region[1], region[2]+padding, region[3])
            self.graph.configure(scrollregion=region)
            if focus:
                x, y = self.positions[event.node_id]
                self.center_graph((x+105)*z, (y+56)*z)

    def center_graph(self, x, y):
        values = self.graph.cget("scrollregion").split()
        if len(values) != 4:
            return
        x0, y0, x1, y1 = map(float, values)
        self.graph.xview_moveto(max(0, x-self.graph.winfo_width()/2-x0)/max(1, x1-x0))
        self.graph.yview_moveto(max(0, y-self.graph.winfo_height()/2-y0)/max(1, y1-y0))

    def zoom_graph(self, event, factor=None):
        if not self.result or self.event_index < 0:
            return "break"
        factor = factor or (1.1 if event.delta>0 else 1/1.1)
        old_zoom = self.zoom
        self.zoom = max(.3, min(1.8, old_zoom*factor))
        cx = self.graph.canvasx(self.graph.winfo_width()/2)/old_zoom
        cy = self.graph.canvasy(self.graph.winfo_height()/2)/old_zoom
        self.draw_search(focus=False)
        self.center_graph(cx*self.zoom, cy*self.zoom)
        return "break"

    def reset_route(self):
        self.pause()
        self.route_index = 0
        self.show_route()

    def previous_route(self):
        self.pause()
        self.route_index = max(0, self.route_index-1)
        self.show_route()

    def manual_route(self):
        self.pause()
        self.next_route()

    def next_route(self):
        if self.result and self.route_index < len(self.result.path)-1:
            self.route_index += 1
            self.show_route()
            return True
        return False

    def show_route(self):
        self.draw_river()
        if self.result and self.result.path:
            iid = str(self.route_index)
            self.route_table.selection_set(iid)
            self.route_table.see(iid)

    def select_route(self, _):
        selection = self.route_table.selection()
        if selection and self.result:
            new_index = int(selection[0])
            if new_index != self.route_index:
                self.pause()
                self.route_index = new_index
                self.draw_river()

    def draw_river(self):
        canvas = self.river
        canvas.delete("all")
        width = max(700, canvas.winfo_width())
        if not self.result or not self.result.path:
            self.route_label_var.set("Sin cruces que mostrar.")
            canvas.create_text(width/2, 130, text="La ruta aparecerá aquí al resolver un caso válido.",
                               fill=self.MUTED, font=(self.font_family, 12))
            return
        r = self.result
        node = r.path[self.route_index]
        state = node.state
        completed = self.route_index==len(r.path)-1
        left_edge, right_edge = width*.36, width*.64
        canvas.create_rectangle(0, 0, left_edge, 265, fill="#1c2520", outline="")
        canvas.create_rectangle(left_edge, 0, right_edge, 265, fill="#142a3a", outline="")
        canvas.create_rectangle(right_edge, 0, width, 265, fill="#1c2520", outline="")
        for center, title, m, c in ((left_edge/2, "IZQUIERDA", state.missionaries_left, state.cannibals_left),
                                  ((right_edge+width)/2, "DERECHA", r.missionaries-state.missionaries_left,
                                   r.cannibals-state.cannibals_left)):
            canvas.create_text(center, 31, text=title, fill=self.MUTED, font=(self.font_family, 10, "bold"))
            for y, label, amount, color in ((85, "Misioneros", m, self.GREEN), (155, "Caníbales", c, self.MAROON)):
                canvas.create_text(center, y, text=f"{label}: {amount}", fill=self.TEXT, font=(self.font_family, 13))
                visible = min(amount, 10)
                for i in range(visible):
                    x = center-(visible-1)*10+i*20
                    canvas.create_oval(x-5, y+19, x+5, y+29,
                                       fill=self.NEON if completed and title=="DERECHA" else color, outline="")
                if amount>10:
                    canvas.create_text(center, y+47, text=f"+ {amount-10}", fill=self.MUTED)
        canvas.create_text(width/2, 32, text="RÍO", fill="#759bad", font=(self.font_family, 10, "bold"))
        for y in (74, 106, 138, 206):
            canvas.create_line(left_edge+25, y, right_edge-25, y, fill="#203f53")
        boat_x = left_edge+62 if state.boat=="I" else right_edge-62
        color = self.NEON if completed else "#779785"
        canvas.create_polygon(boat_x-39, 158, boat_x+39, 158, boat_x+26, 181,
                              boat_x-26, 181, fill="#385048", outline=color, width=2)
        canvas.create_line(boat_x+7, 151, boat_x+25, 132, fill=color, width=3)
        canvas.create_text(boat_x, 225, text=f"Barca en {state.boat}", fill=self.TEXT, font=(self.font_family, 10))
        description = str(node.action) if node.action else "Estado inicial"
        self.route_label_var.set(f"Cruce {self.route_index} de {len(r.path)-1}  ·  {description}  ·  Estado {state}")

    def toggle_play(self, mode):
        if self.playing:
            self.pause()
            return
        if not self.result:
            return
        if mode=="route" and self.route_index==len(self.result.path)-1:
            self.reset_route()
        self.playing = True
        button = self.search_play if mode=="search" else self.route_play
        button.configure(text="Pausar")

        def tick():
            self.timer = None
            if not self.playing:
                return
            advanced = self.next_event() if mode=="search" else self.next_route()
            if not advanced or not self.playing:
                self.pause()
                return
            self.timer = self.root.after(550, tick)

        tick()

    def pause(self):
        self.playing = False
        if self.timer is not None:
            self.root.after_cancel(self.timer)
            self.timer = None
        if hasattr(self, "search_play"):
            self.search_play.configure(text="Reproducir")
        if hasattr(self, "route_play"):
            self.route_play.configure(text="Reproducir")

    def close(self):
        self.pause()
        self.root.destroy()


def print_result(result: SearchResult):
    print(result.message)
    print(f"Capacidad: {result.capacity if result.capacity is not None else 'no aplicable'}")
    for step, node in enumerate(result.path):
        print(f"{step:2} | {str(node.action) if node.action else 'Inicio':20} | "
              f"{node.state} | g={node.g}, h={node.h}, f={node.f}")
    s = result.metrics
    cost = calculate_costs(result)
    print(f"CR={cost.route if cost.route is not None else 'sin ruta'} cruces; "
          f"CT={cost.time}; CE={cost.space} (CT=CE=expansiones, según la práctica)")
    print(f"CB={cost.search:g} puntos; Total={format(cost.total, 'g') if cost.total is not None else 'no aplicable'} "
          f"(índice: {ROUTE_WEIGHT:g}×CR + {TIME_WEIGHT:g}×CT + {SPACE_WEIGHT:g}×CE)")
    print(f"Descubiertos={s.discovered}; expandidos={s.expanded}; sucesores={s.successors}; "
          f"descartados={s.discarded}; mejoras={s.improvements}; frontera máxima={s.frontier_peak}")
    print(f"Tiempo de búsqueda: {s.elapsed_ms:.3f} ms")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--consola", action="store_true", help="Resolver sin interfaz gráfica")
    parser.add_argument("-m", "--misioneros", type=int, default=3)
    parser.add_argument("-c", "--canibales", type=int, default=3)
    args = parser.parse_args()
    if args.consola:
        try:
            result = solve(args.misioneros, args.canibales)
        except ValueError as error:
            parser.error(str(error))
        print_result(result)
        return 0 if result.status == "resuelto" else 1
    try:
        import tkinter as tk
        root = tk.Tk()
    except ImportError:
        print("Esta instalación de Python no incluye Tkinter. Prueba --consola o instala Python con Tcl/Tk.")
        return 2
    except tk.TclError as error:
        print(f"No se pudo abrir la ventana: {error}\nPuedes utilizar --consola.")
        return 2
    app = Application(root)
    app.m_var.set(str(args.misioneros))
    app.c_var.set(str(args.canibales))
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
