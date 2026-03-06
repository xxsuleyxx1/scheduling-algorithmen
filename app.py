"""
Scheduling-Algorithmen: List Scheduling & LPT (Longest Processing Time)
========================================================================
Interaktive Streamlit-Anwendung zur Visualisierung und Analyse von
Scheduling-Algorithmen fuer das Makespan-Minimierungsproblem (P||Cmax).

Verbesserungen gegenueber der vorherigen Version:
- LPT nutzt ListScheduling als Subroutine (keine Codeduplizierung)
- Effiziente Lastberechnung ueber separate Lastliste (kein wiederholtes sum())
- Speicherverbrauch wird gemessen und visualisiert
- Ausfuehrliche Dokumentation und Kommentare
"""

import streamlit as st
import yaml
import timeit
import tracemalloc
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO

TIMING_RUNS = 500


def _scheduling_kernel(jobs, num_machines):
    """
    Interner Scheduling-Kern ohne Zeitmessung.

    Enthaelt die reine Job-Zuteilungslogik, die von list_scheduling und
    (indirekt ueber list_scheduling) von list_scheduling_lpt genutzt wird.
    Trennt die Algorithmus-Logik sauber von der Laufzeitmessung.

    Parameter:
        jobs (list):        Joblaengen in der gewuenschten Verarbeitungsreihenfolge.
        num_machines (int): Anzahl der Maschinen.

    Rueckgabe:
        tuple: (machines, machine_loads)
    """
    machines = [[] for _ in range(num_machines)]
    machine_loads = [0] * num_machines
    for job in jobs:
        min_index = machine_loads.index(min(machine_loads))
        machines[min_index].append(job)
        machine_loads[min_index] += job
    return machines, machine_loads


def list_scheduling(jobs, num_machines):
    """
    List-Scheduling-Algorithmus fuer das Makespan-Minimierungsproblem.

    Weist jeden Job der Maschine zu, die aktuell die geringste Last hat.
    Delegiert die Zuteilungslogik an _scheduling_kernel.

    Die Laufzeit wird als Mittelwert ueber TIMING_RUNS Wiederholungen
    gemessen (timeit), um Messrauschen durch das Betriebssystem zu
    glaetten und stabile, reproduzierbare Ergebnisse zu erzielen.

    Parameter:
        jobs (list[int/float]): Liste der Joblaengen in der Reihenfolge,
                                in der sie zugewiesen werden sollen.
        num_machines (int):     Anzahl der verfuegbaren Maschinen.

    Rueckgabe:
        tuple: (machines, machine_loads, runtime_seconds, peak_memory_bytes)
    """
    machines, machine_loads = _scheduling_kernel(jobs, num_machines)

    total_time = timeit.timeit(
        lambda: _scheduling_kernel(jobs, num_machines),
        number=TIMING_RUNS
    )
    runtime = total_time / TIMING_RUNS

    tracemalloc.start()
    _scheduling_kernel(jobs, num_machines)
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return machines, machine_loads, runtime, peak_memory


def list_scheduling_lpt(jobs, num_machines):
    """
    LPT-Algorithmus (Longest Processing Time first).

    Sortiert die Jobs absteigend nach ihrer Laenge und ruft list_scheduling
    als Subroutine auf, um die eigentliche Zuteilung durchzufuehren.

    Die Laufzeitmessung umfasst sowohl die Sortierung (O(n log n)) als auch
    die Scheduling-Phase (via _scheduling_kernel), sodass LPT immer eine
    hoehere oder gleiche gemessene Laufzeit als List Scheduling hat.
    Auch hier wird der Mittelwert ueber TIMING_RUNS Wiederholungen verwendet.

    Parameter:
        jobs (list[int/float]): Unsortierte Liste der Joblaengen.
        num_machines (int):     Anzahl der verfuegbaren Maschinen.

    Rueckgabe:
        tuple: (machines, machine_loads, runtime_seconds, peak_memory_bytes)
               Gleiche Struktur wie list_scheduling().
    """
    sorted_jobs = sorted(jobs, reverse=True)
    machines, machine_loads = list_scheduling(sorted_jobs, num_machines)[:2]

    def _lpt_run():
        sj = sorted(jobs, reverse=True)
        _scheduling_kernel(sj, num_machines)

    total_time = timeit.timeit(_lpt_run, number=TIMING_RUNS)
    runtime = total_time / TIMING_RUNS

    tracemalloc.start()
    _lpt_run()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return machines, machine_loads, runtime, peak_memory


def parse_yaml_instances(file_content):
    """
    Liest YAML-Instanzen aus einem String und gibt sie als Liste zurueck.

    Jede Instanz muss die Schluessel 'id', 'num_machines' und 'jobs' enthalten.

    Parameter:
        file_content (str): Inhalt einer YAML-Datei mit mehreren Dokumenten
                            (getrennt durch '---').

    Rueckgabe:
        list[dict]: Liste der geparsten Instanzen.
    """
    return list(yaml.safe_load_all(file_content))


def create_comparison_charts(instance_id, ls_result, lpt_result, num_machines):
    """
    Erstellt eine Vergleichsvisualisierung der Ergebnisse von
    List Scheduling und LPT fuer eine gegebene Instanz.

    Erzeugt ein 2x3-Diagrammgitter mit:
      - Zeile 1: Maschinenbelastung, Laufzeit, Speicherverbrauch fuer LS
      - Zeile 2: Maschinenbelastung, Laufzeit, Speicherverbrauch fuer LPT

    Parameter:
        instance_id:   ID der Instanz (fuer Titel).
        ls_result:     Ergebnis-Tuple von list_scheduling().
        lpt_result:    Ergebnis-Tuple von list_scheduling_lpt().
        num_machines:  Anzahl der Maschinen.

    Rueckgabe:
        plotly.graph_objects.Figure: Das fertige Vergleichsdiagramm.
    """
    ls_machines, ls_loads, ls_time, ls_mem = ls_result
    lpt_machines, lpt_loads, lpt_time, lpt_mem = lpt_result

    machine_labels = [f"M{i+1}" for i in range(num_machines)]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            "Maschinenbelastung (List Scheduling)",
            "Laufzeit (List Scheduling)",
            "Speicherverbrauch (List Scheduling)",
            "Maschinenbelastung (LPT)",
            "Laufzeit (LPT)",
            "Speicherverbrauch (LPT)",
        ),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    fig.add_trace(
        go.Bar(x=machine_labels, y=ls_loads, marker_color="skyblue", name="LS Last",
               text=ls_loads, textposition="outside"),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(x=["Laufzeit"], y=[ls_time], marker_color="orange", name="LS Zeit",
               text=[f"{ls_time:.6f} s"], textposition="outside"),
        row=1, col=2,
    )
    fig.add_trace(
        go.Bar(x=["Speicher"], y=[ls_mem / 1024], marker_color="salmon", name="LS Speicher",
               text=[f"{ls_mem / 1024:.2f} KB"], textposition="outside"),
        row=1, col=3,
    )

    fig.add_trace(
        go.Bar(x=machine_labels, y=lpt_loads, marker_color="lightgreen", name="LPT Last",
               text=lpt_loads, textposition="outside"),
        row=2, col=1,
    )
    fig.add_trace(
        go.Bar(x=["Laufzeit"], y=[lpt_time], marker_color="green", name="LPT Zeit",
               text=[f"{lpt_time:.6f} s"], textposition="outside"),
        row=2, col=2,
    )
    fig.add_trace(
        go.Bar(x=["Speicher"], y=[lpt_mem / 1024], marker_color="mediumpurple", name="LPT Speicher",
               text=[f"{lpt_mem / 1024:.2f} KB"], textposition="outside"),
        row=2, col=3,
    )

    fig.update_yaxes(title_text="Gesamtbearbeitungszeit", row=1, col=1)
    fig.update_yaxes(title_text="Sekunden", row=1, col=2)
    fig.update_yaxes(title_text="KB", row=1, col=3)
    fig.update_yaxes(title_text="Gesamtbearbeitungszeit", row=2, col=1)
    fig.update_yaxes(title_text="Sekunden", row=2, col=2)
    fig.update_yaxes(title_text="KB", row=2, col=3)

    fig.update_layout(
        height=650,
        title_text=f"Vergleich: List Scheduling vs. LPT — Instanz {instance_id}",
        showlegend=False,
        uniformtext_minsize=8,
        uniformtext_mode="hide",
    )

    return fig


def create_makespan_comparison(instance_id, ls_loads, lpt_loads):
    """
    Erstellt ein Balkendiagramm, das den Makespan (maximale Maschinenlast)
    beider Algorithmen direkt vergleicht.

    Parameter:
        instance_id:  ID der Instanz.
        ls_loads:     Lastliste von List Scheduling.
        lpt_loads:    Lastliste von LPT.

    Rueckgabe:
        plotly.graph_objects.Figure: Makespan-Vergleichsdiagramm.
    """
    ls_makespan = max(ls_loads)
    lpt_makespan = max(lpt_loads)

    fig = go.Figure(data=[
        go.Bar(
            x=["List Scheduling", "LPT"],
            y=[ls_makespan, lpt_makespan],
            marker_color=["skyblue", "lightgreen"],
            text=[ls_makespan, lpt_makespan],
            textposition="auto",
        )
    ])

    fig.update_layout(
        title=f"Makespan-Vergleich — Instanz {instance_id}",
        yaxis_title="Makespan (Cmax)",
        height=350,
    )

    return fig


st.set_page_config(page_title="Scheduling-Algorithmen", layout="wide")
st.title("Scheduling-Algorithmen: List Scheduling & LPT")
st.markdown(
    "Vergleich von **List Scheduling** und **LPT** (Longest Processing Time) "
    "fuer das Makespan-Minimierungsproblem P||Cmax."
)

st.divider()
st.subheader("Theoretischer Hintergrund")

tab_prob, tab_ls_theo, tab_lpt_theo = st.tabs([
    "Problemstellung", "List Scheduling", "LPT"
])

with tab_prob:
    st.markdown("### Das Problem P||Cmax")
    st.markdown(
        "Wir betrachten das klassische Scheduling-Problem **P||Cmax** "
        "aus der Kombinatorischen Optimierung:"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Gegeben:**")
        st.markdown(
            "- **n Jobs** mit unterschiedlichen Bearbeitungszeiten\n"
            "- **m identische, parallele Maschinen**\n"
            "- Jeder Job wird genau einer Maschine zugewiesen\n"
            "- Keine Unterbrechung (nicht unterbrechbar)"
        )
    with col2:
        st.markdown("**Gesucht:**")
        st.markdown(
            "Zuweisung aller Jobs auf die Maschinen, sodass der **Makespan Cmax "
            "minimiert** wird."
        )
    st.markdown("**Definition Makespan:**")
    st.latex(r"C_{\max} = \max_{i=1}^{m} \sum_{j \in M_i} p_j")
    st.markdown(
        "Das ist der Zeitpunkt, zu dem die **letzte Maschine ihre Arbeit beendet** — "
        "also wann wirklich alles erledigt ist."
    )
    st.divider()
    st.markdown("### Warum ist das Problem schwer?")
    st.markdown(
        "- Das Problem P||Cmax ist **NP-schwer im starken Sinne** (selbst fuer m = 2).\n"
        "- Die Anzahl moeglicher Zuweisungen waechst **exponentiell** mit der Anzahl der Jobs: "
        "fuer n Jobs auf m Maschinen gibt es m^n moegliche Verteilungen.\n"
        "- Es ist kein Algorithmus bekannt, der fuer grosse Instanzen in polynomieller Zeit "
        "garantiert die optimale Loesung berechnet.\n"
        "- **Loesung:** Approximationsalgorithmen, die in kurzer Laufzeit eine Loesung nahe "
        "am Optimum liefern und zugleich eine theoretische **Guetegarantie** besitzen."
    )

with tab_ls_theo:
    st.markdown("### List Scheduling (LS)")
    st.markdown(
        "Der List-Scheduling-Algorithmus geht auf **R.L. Graham (1969)** zurueck "
        "und war einer der ersten Algorithmen mit nachgewiesener Approximationsguete."
    )
    st.markdown("**Idee (Greedy-Strategie):**")
    st.markdown(
        "1. Gehe die Jobs in der **gegebenen (beliebigen) Reihenfolge** durch.\n"
        "2. Weise jeden Job der Maschine zu, die aktuell die **kleinste Gesamtlast** hat.\n"
        "3. Keine Vorausschau — es wird immer die **lokal beste Entscheidung** getroffen."
    )
    st.divider()
    st.markdown("**Laufzeitkomplexitaet:**")
    st.latex(r"O(n \cdot m)")
    st.markdown(
        "Fuer jeden der n Jobs wird unter m Maschinen die freiste gesucht. "
        "Bei unserem Ansatz mit einer separaten Lastliste ist das Minimum in O(m) findbar."
    )
    st.divider()
    st.markdown("**Approximationsgarantie (Theorem 8.1, Graham 1969):**")
    st.latex(r"\frac{C_{\max}(\text{LS})}{C_{\max}^*} \leq 2 - \frac{1}{m}")
    st.markdown(
        "Die gefundene Loesung ist **hoechstens (2 - 1/m)-mal so schlecht** wie die "
        "optimale Loesung. Fuer grosse m naehert sich der Faktor dem Wert **2** an.\n\n"
        "**Intuition:** Sei Job l der zuletzt fertige Job. Vor seinem Start ist keine Maschine "
        "unbeschaeftigt — daher ist die Summe aller anderen Jobs mindestens m-mal die Startzeit von l. "
        "Daraus folgt die Schranke direkt."
    )

with tab_lpt_theo:
    st.markdown("### LPT — Longest Processing Time")
    st.markdown(
        "LPT ist eine **Verbesserung von List Scheduling** durch eine einfache, "
        "aber wirkungsvolle Idee: die Sortierung der Jobs vor der Zuweisung."
    )
    st.markdown("**Idee:**")
    st.markdown(
        "1. **Sortiere** die Jobs in **absteigender Reihenfolge** nach ihrer Bearbeitungszeit.\n"
        "2. Wende danach **List Scheduling als Subroutine** an.\n\n"
        "Grosse Jobs haben den groessten Einfluss auf den Makespan. "
        "Werden sie zuerst verteilt, entsteht eine gleichmaessigere Balance — "
        "spaete 'Ueberladungen' einer Maschine werden vermieden."
    )
    st.divider()
    st.markdown("**Laufzeitkomplexitaet:**")
    st.latex(r"O(n \log n + n \cdot m)")
    st.markdown(
        "- O(n log n) fuer die absteigende Sortierung der Jobs\n"
        "- O(n · m) fuer die anschliessende List-Scheduling-Phase\n\n"
        "LPT benoetigt etwas mehr **Speicher** als LS, da eine sortierte Kopie "
        "der Jobliste angelegt wird."
    )
    st.divider()
    st.markdown("**Approximationsgarantie (Theorem 8.2, Graham 1969):**")
    st.latex(r"\frac{C_{\max}(\text{LPT})}{C_{\max}^*} \leq \frac{4}{3} - \frac{1}{3m}")
    st.markdown(
        "Deutlich bessere Garantie als List Scheduling! "
        "Fuer grosse m naehert sich der Faktor dem Wert **4/3 ≈ 1.33** an.\n\n"
        "**Spezialfall:** Wenn der letzte Job kuerzere Laufzeit als Cmax*/3 hat, "
        "folgt die Schranke direkt aus der LS-Analyse. "
        "Andernfalls liefert LPT sogar die **optimale Loesung**."
    )
    st.divider()
    st.markdown("### Vergleich der Algorithmen")
    st.markdown(
        "| Algorithmus | Strategie | Laufzeit | Approximationsgarantie |\n"
        "|---|---|---|---|\n"
        "| List Scheduling | Greedy, beliebige Reihenfolge | O(n · m) | ≤ 2 − 1/m |\n"
        "| LPT | Greedy, absteigende Sortierung | O(n log n + n · m) | ≤ 4/3 − 1/(3m) |\n"
    )
    st.info(
        "Beide Algorithmen sind einfach zu implementieren und in der Praxis sehr effizient. "
        "LPT liefert sowohl theoretisch als auch praktisch bessere Ergebnisse "
        "bei nur geringem Mehraufwand."
    )

st.divider()
st.sidebar.header("Instanz laden")

source = st.sidebar.radio(
    "Datenquelle waehlen:",
    ["medium_instances.yaml", "Eigene YAML-Datei hochladen"],
)

data = None

if source == "medium_instances.yaml":
    try:
        with open("medium_instances.yaml", "r") as f:
            data = parse_yaml_instances(f.read())
    except FileNotFoundError:
        st.error("Datei 'medium_instances.yaml' nicht gefunden.")
else:
    uploaded = st.sidebar.file_uploader("YAML-Datei hochladen", type=["yaml", "yml"])
    if uploaded is not None:
        content = StringIO(uploaded.getvalue().decode("utf-8")).read()
        try:
            data = parse_yaml_instances(content)
        except yaml.YAMLError as e:
            st.error(f"Fehler beim Parsen der YAML-Datei: {e}")

if data:
    instance_options = {f"Instanz {d['id']}": i for i, d in enumerate(data)}
    selected_label = st.sidebar.selectbox("Instanz auswaehlen:", list(instance_options.keys()))
    selected_index = instance_options[selected_label]

    instance = data[selected_index]
    instance_id = instance["id"]
    num_machines = instance["num_machines"]
    jobs = instance["jobs"]

    st.subheader(f"Instanz {instance_id}")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.metric("Anzahl Maschinen", num_machines)
    with col_info2:
        st.metric("Anzahl Jobs", len(jobs))
    st.write("**Jobs:**", jobs)

    if st.button("Algorithmen ausfuehren", type="primary"):
        with st.spinner("Berechnung laeuft..."):
            ls_result = list_scheduling(jobs, num_machines)
            lpt_result = list_scheduling_lpt(jobs, num_machines)

        ls_machines, ls_loads, ls_time, ls_mem = ls_result
        lpt_machines, lpt_loads, lpt_time, lpt_mem = lpt_result

        st.subheader("Ergebnisse")

        tab_ls, tab_lpt = st.tabs(["List Scheduling", "LPT"])

        with tab_ls:
            st.markdown("**Maschinenzuweisungen:**")
            df_ls = pd.DataFrame({
                "Maschine": [f"Maschine {i+1}" for i in range(num_machines)],
                "Zugewiesene Jobs": [str(m) for m in ls_machines],
                "Gesamtlast": ls_loads,
            })
            st.dataframe(df_ls, use_container_width=True, hide_index=True)
            st.write(f"**Makespan (Cmax):** {max(ls_loads)}")
            st.write(f"**Laufzeit:** {ls_time:.6f} Sekunden")
            st.write(f"**Speicherverbrauch:** {ls_mem / 1024:.2f} KB")

        with tab_lpt:
            st.markdown("**Maschinenzuweisungen (Jobs absteigend sortiert):**")
            df_lpt = pd.DataFrame({
                "Maschine": [f"Maschine {i+1}" for i in range(num_machines)],
                "Zugewiesene Jobs": [str(m) for m in lpt_machines],
                "Gesamtlast": lpt_loads,
            })
            st.dataframe(df_lpt, use_container_width=True, hide_index=True)
            st.write(f"**Makespan (Cmax):** {max(lpt_loads)}")
            st.write(f"**Laufzeit:** {lpt_time:.6f} Sekunden")
            st.write(f"**Speicherverbrauch:** {lpt_mem / 1024:.2f} KB")

        st.subheader("Visualisierung")
        fig_compare = create_comparison_charts(instance_id, ls_result, lpt_result, num_machines)
        st.plotly_chart(fig_compare, use_container_width=True)

        fig_makespan = create_makespan_comparison(instance_id, ls_loads, lpt_loads)
        st.plotly_chart(fig_makespan, use_container_width=True)

        ls_makespan = max(ls_loads)
        lpt_makespan = max(lpt_loads)
        if lpt_makespan < ls_makespan:
            verbesserung = ((ls_makespan - lpt_makespan) / ls_makespan) * 100
            st.success(
                f"LPT verbessert den Makespan um {verbesserung:.1f}% "
                f"gegenueber List Scheduling ({ls_makespan} -> {lpt_makespan})."
            )
        elif lpt_makespan == ls_makespan:
            st.info(
                f"Beide Algorithmen liefern den gleichen Makespan: {ls_makespan}."
            )
        else:
            st.warning(
                f"List Scheduling liefert hier einen besseren Makespan "
                f"({ls_makespan} vs. {lpt_makespan})."
            )
else:
    st.info("Bitte eine Instanz-Datei laden, um die Algorithmen auszufuehren.")
