# Carnot Web Frontend

React frontend for the Carnot Web application.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm run dev
```

The app will be available at http://localhost:5173

## Build for Production

```bash
npm run build
```

## Project Structure

```
src/
├── components/
│   ├── Layout.jsx              # Main layout with navigation
│   └── DatasetCreator/
│       ├── FileBrowser.jsx     # File system browser
│       ├── SearchChatbot.jsx   # AI-powered file search
│       └── DatasetAnnotation.jsx # Dataset metadata form
├── pages/
│   ├── DataManagementPage.jsx  # Main page for datasets and uploads
│   ├── DatasetCreatorPage.jsx  # Create new datasets
│   └── UserChatPage.jsx        # Chat interface (placeholder)
├── services/
│   └── api.js                  # API client
├── App.jsx                     # Main app component with routing
├── main.jsx                    # Entry point
└── index.css                   # Global styles
```

## Features

### Data Management Page
- View all created datasets
- Upload new files
- Delete datasets
- Navigate to dataset creator

### Dataset Creator Page
- Browse files and directories (like macOS Finder)
- AI-powered file search chatbot
- Select files with checkboxes
- Add dataset name and annotations
- Save dataset

### User Chat Page
- Placeholder for future chat functionality

