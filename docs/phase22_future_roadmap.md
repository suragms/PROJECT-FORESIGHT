# Phase 22 — Future Roadmap

Items below are **planned improvements**, not implemented features.

## Short Term

- **Collect production actuals** — compare live demand to forecasts
- **Measure live WAPE** — replace PENDING_ACTUALS with measured performance
- **Monitor holiday periods** — track Nov–Dec bias in production data
- **Alert response playbook** — document operator actions for each alert type

## Medium Term

- **Improve holiday modelling** — additional calendar features, event-specific adjustments
- **Evaluate quantile forecasts** — P10/P90 intervals for uncertainty-aware planning
- **Evaluate intermittent-demand methods** — hurdle/Croston for sparse SKUs
- **Expand risk matrix** — beyond 100-SKU reference extract

## Long Term

- **Automated retraining governance** — controlled promotion pipeline with drift triggers
- **Advanced drift response** — investigation workflows, not automatic replacement
- **Multi-source inventory integration** — connect to live WMS/ERP feeds
- **Cloud deployment** — TLS, identity provider, secrets manager, autoscaling

## Explicitly Not Implemented

- Cloud production deployment
- Automated model replacement
- Guaranteed financial ROI measurement
- Quantile/hurdle production models for Phase 20 path
