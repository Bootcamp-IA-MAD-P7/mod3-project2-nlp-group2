# YouTube Comment Moderator (frontend)

A web app for moderating YouTube comments. The moderator pastes a video link, the app loads comments already classified by a toxicity model, and shows them as swipeable cards: swipe left to not publish, swipe right to publish.

## Tech stack

- React 19 and TypeScript
- Vite (dev server and build)
- framer-motion (card drag and swipe animation)
- ESLint (linting)

## Run locally

Requires Node.js and npm (`package.json` does not pin a Node version).

```bash
cd frontend
npm install
npm run dev
```

Vite prints the local address, normally `http://localhost:5173`.

Other scripts: `npm run build`, `npm run lint`, `npm run preview`.

## Backend connection

The app does not classify anything itself. It calls a separate FastAPI backend, expected at `http://localhost:8000` (set in `src/lib/api.ts`), using two endpoints: `GET /comments/video` and `GET /comments/comment`. Without the backend running, no comments load; errors only show in the browser console.

Swipe decisions are kept in memory only. The app does not send them to the backend yet.

## Project structure

```
src/
  main.tsx                    App entry point
  App.tsx                     Holds the comment queue and handles each action
  types.ts                    Shared types (comment, decision)
  components/
    Sidebar.tsx               Video link box, single-comment lookup, "Load more"
    Screen.tsx                Main stage: background, hint text, left/right labels
    ModerationDeck.tsx        Stacks the next 3 cards
    Card.tsx                  One swipeable comment card (framer-motion)
    Side.tsx                  The "Don't publish" / "Publish" arrow labels
  lib/
    api.ts                    Calls to the backend
    youtube.ts                Gets video and comment ids from YouTube links
```
