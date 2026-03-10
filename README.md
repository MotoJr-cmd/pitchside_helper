## Pitchside Helper HUD

Pitchside Helper HUD is an open-source, AI-driven tactical analysis tool designed to bridge the gap between “hard-coded” statistical grading and the actual tactical flow of a football match.

While traditional platforms often penalize players based on isolated events (e.g., “Big Chance Missed”), Pitchside Helper utilizes real-time computer vision and advanced data scrapers to provide a context-aware evaluation of player performance.

## The vision

Traditional soccer grading often lacks context. A striker missing a 0.8 xG chance is frequently penalized in ratings regardless of defensive pressure, body shape, or the difficulty of the pass received.

Pitchside Helper aims to be a “Statecast for the Everyman,” providing:

- **Real-time tracking**: Leveraging YOLO11 to monitor player positioning and “zonal dominance”.
- **Contextual grading**: Comparing live visual data against historical advanced metrics (xG, xT, progressive passes) from FBref and SofaScore.
- **Floating HUD**: A sleek, transparent interface that overlays directly onto your match stream for an “Invisible Helper” experience.

## Tech stack

- **Language**: Python 3.10+
- **Computer vision**: YOLO11 (via Ultralytics) for object detection and player tracking
- **Data acquisition**: SoccerData library for multi-source scraping (FBref, SofaScore, WhoScored)
- **GUI / interface**: CustomTkinter for a high-performance, semi-transparent HUD
- **AI logic**: Integration with LLMs for natural language tactical commentary

## Roadmap

- [ ] **Phase 1**: The Gap Analyzer — scripting the comparison between “harsh” ratings and “advanced” impact stats
- [ ] **Phase 2**: Visual Intelligence — implementing YOLO11 for live player and ball tracking
- [ ] **Phase 3**: The Overlay — launching the borderless, transparent HUD for real-time match viewing
- [ ] **Phase 4**: Predictive Analytics — real-time next-play probability based on player positioning

## License

This project is licensed under the GNU GPLv3.

Pitchside Helper is intended for educational and personal research purposes. If you use this code in your own project, you must keep it open-source and provide attribution to the original author.
