# Barista Agent

An AI-powered coffee shop ordering assistant built with LangGraph and Gemini.

## Architecture

```
┌─────────────────────────┐      ┌─────────────────────────┐
│  Next.js Frontend       │ ──── │  FastAPI Backend        │
│  (Vercel)               │      │  (Railway)              │
└─────────────────────────┘      └─────────────────────────┘
                                          │
                                 ┌────────┴────────┐
                                 │  LangGraph      │
                                 │  Agent          │
                                 └────────┬────────┘
                                          │
                                 ┌────────┴────────┐
                                 │  Gemini 2.0     │
                                 │  Flash          │
                                 └─────────────────┘
```

## Features

- Conversational ordering experience
- Menu browsing with prices
- Order management (add, view, clear items)
- Price calculation with modifiers
- Order confirmation flow

## Tech Stack

- **Frontend**: Next.js 14, React, Tailwind CSS
- **Backend**: FastAPI, Python 3.11+
- **AI**: LangGraph, Google Gemini 2.0 Flash
- **Deployment**: Vercel (frontend), Railway (backend)

## Local Development

### Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
cp .env.example .env.local
# Update NEXT_PUBLIC_API_URL if needed
npm run dev
```

## Deployment

### Backend (Railway)

1. Create a new project on [Railway](https://railway.app)
2. Connect your GitHub repo
3. Set root directory to `/backend`
4. Add environment variable: `GOOGLE_API_KEY`
5. Railway auto-detects the Dockerfile

### Frontend (Vercel)

1. Import project on [Vercel](https://vercel.com)
2. Set root directory to `/frontend`
3. Add environment variable: `NEXT_PUBLIC_API_URL` (your Railway backend URL)
4. Deploy

## LangGraph Agent Design

The agent uses a state machine with these components:

**State**:
- `messages`: Conversation history
- `order`: Current order items
- `finished`: Order completion flag

**Tools**:
- `get_menu()`: Display menu
- `add_to_order(item)`: Add item to order
- `get_order()`: Show current order
- `confirm_order()`: Review order before placing
- `place_order()`: Finalize order
- `calculate_total()`: Show price breakdown

**Flow**:
```
START → barista → [tool calls?]
                      ↓
            ┌────────┴────────┐
            │                 │
        order_node          tools
            │                 │
            └────────┬────────┘
                     ↓
                  barista → END (if no tools)
```

## License

MIT
