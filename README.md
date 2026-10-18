# Direct Multi-Horizon GATv2 for Train Delay Prediction

**Thinh Duy Do** · Halıcıoğlu Data Science Institute, UC San Diego

📄 [Paper](paper/direc_multihorizon_GATv2.pdf) · 🖼️ [Poster](poster/conference_poster.pdf) · 🎤 [Slides](poster/conference_slides_Thinh_Duy_Do.pdf)

Graph neural network forecasting of NYC subway delays. This project asks whether **predicting several future delays at once (direct multi-horizon)** reduces the error build-up that happens when a model **feeds its own predictions back in (autoregressive)**, using the same event-based graph and GATv2 encoder for both.

---

## Key results

Validation set, 77,361 events. MAE and RMSE in minutes.

| Model                       | MAE       | RMSE      | Accuracy  | Precision | Recall    | F1        |
| --------------------------- | --------- | --------- | --------- | --------- | --------- | --------- |
| Zero-delay                  | 6.710     | 13.711    | 0.517     | 0.000     | 0.000     | 0.000     |
| Persistence                 | **0.729** | **1.893** | 0.929     | 0.938     | 0.913     | 0.926     |
| Autoregressive GATv2        | 3.124     | 7.597     | 0.922     | 0.938     | 0.899     | 0.918     |
| **DMH GATv2 + aggregation** | 2.560     | 6.914     | **0.946** | **0.949** | **0.938** | **0.944** |

**Error across horizons** (MAE, minutes):

| Model                | t+1       | t+2       | t+3       | Growth t+1 → t+3 |
| -------------------- | --------- | --------- | --------- | ---------------- |
| Autoregressive GATv2 | **2.411** | 2.861     | 3.124     | +0.71            |
| DMH GATv2            | 2.645     | **2.703** | **2.751** | **+0.11**        |

**Takeaways**

- **DMH error stays nearly flat across horizons**, while autoregressive error grows. That supports the hypothesis that recursive decoding drives error accumulation.
- **DMH is the best delay detector:** highest accuracy, recall and F1 of all models.
- **Persistence is still the strongest regressor.** Subway delays rarely jump between consecutive events, so "the delay stays the same" is a hard baseline to beat on MAE.

---

## Method

**1. Event graph.** Every train **arrival** and **departure** is a node. Edges encode how delays propagate:

- **Dwell:** arrival → departure at the same stop
- **Run:** departure → next arrival of the same train
- **Headway:** consecutive trains on the same route, direction, stop and event type

**2. Features.** Dwell time, running time, headway, trip progress (stop sequence, normalized progress, stops remaining), and time of day (sine/cosine encoding plus a rush-hour flag).

**3. Model.** A 3-layer **GATv2** encoder (Brody et al., 2022) learns attention over neighboring events, followed by a **two-stage hurdle model**: a classifier predicts _whether_ an event is delayed, and a regressor predicts _how much_. The predicted delay is 0 when P(delay) < 0.5.

**4. Forecasting strategies compared**

- **Autoregressive:** predict t+1, feed it back, then predict t+2 and t+3.
- **Direct multi-horizon (DMH):** three classifier and regressor heads predict t+1, t+2 and t+3 at once from the current graph state. When an event receives predictions from several source events, they are combined with weights proportional to 1 / validation MAE of each horizon.

Builds on the event-based GATv2 framework of Nguyen & Li (GeoAI '25).

---

## Data

- **Observed trips:** NYC subway real-time observations from [Subway Data NYC](https://subwaydata.nyc/)
- **Schedule:** official MTA static GTFS

Raw and processed data are not included because of their size. Use `scripts/download_subwaydata_range.py` to download a date range.

---

## Repository structure

```
paper/                 Paper (PDF)
poster/                Conference poster, slides, and HTML poster
scripts/               Data download and clean-dataset build
src/
  data_processing/     Trip matching, event building, cleaning, feature engineering
  graph/               Event graph construction (Dwell/Run/Headway edges) → event_graph_arr_dep.pt
  baslines/            Zero-delay and persistence baselines
  models/              gatv2_ar.py (autoregressive), gatv2_dmh.py (direct multi-horizon)
  training/            train_ar.py, train_dmh.py
  evaluation/          Horizon metrics, event aggregation, final model comparison
results/               Validation metrics and per-event outputs (CSV)
```

## Pipeline

1. `scripts/download_subwaydata_range.py`: download observed subway data
2. `scripts/build_clean_dataset.py`: merge with the GTFS schedule and clean
3. `src/data_processing/`: match trips, build arrival/departure events, engineer features
4. `src/graph/build_graph.py`: build the event graph with Dwell, Run and Headway edges (max headway link 30 min), saved as `event_graph_arr_dep.pt`
5. `src/baslines/evaluate_baselines.py`: zero-delay and persistence baselines
6. `src/training/train_ar.py` and `src/training/train_dmh.py`: train both GATv2 variants on the event graph
7. `src/evaluation/compare_final_models.py`: produce the comparison tables

**Main libraries:** Python, pandas, NumPy, PyTorch, PyTorch Geometric

---

## Limitations and future work

- Only a fixed 3-event horizon was tested. Other horizon lengths may change the comparison.
- The learned models do not yet beat persistence on MAE. Systems with more sudden disruptions and congestion may show a larger benefit from topology-aware models.
- Next steps: vary horizon length, test on other rail networks, and use forecasts for operational decision support.

## References

1. V. D. A. Nguyen and Z. Li, "Deep Learning to Identify the Spatio-Temporal Cascading Effects of Train Delays in a High-Density Network," GeoAI '25, 2025.
2. S. Brody, U. Alon, and E. Yahav, "How Attentive Are Graph Attention Networks?" ICLR, 2022.
3. T. N. Kipf and M. Welling, "Semi-Supervised Classification with Graph Convolutional Networks," ICLR, 2017.
4. T. Spanninger et al., "A review of train delay prediction approaches," Journal of Rail Transport Planning & Management, 2022.
