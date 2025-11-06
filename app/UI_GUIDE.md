# Carnot Web - UI Guide

This guide provides an overview of the user interface and user experience flows.

## 🎨 Visual Layout

### Header & Navigation
```
┌─────────────────────────────────────────────────────────────────┐
│ Carnot Web Interface                                            │
├─────────────────────────────────────────────────────────────────┤
│  [Data Management]  [User Chat]                                 │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Page Layouts

### 1. Data Management Page

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Data Management                          [+ Create Dataset]    │
│  Manage your datasets and upload files                          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📤 Upload Files                                                 │
│  ┌───────────────────────────────────────────────────────────┐ │
│  │                                                             │ │
│  │              📁                                             │ │
│  │         Click to upload a file                              │ │
│  │         or drag and drop                                    │ │
│  │                                                             │ │
│  └───────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Recently Uploaded (5)                                          │
│  • document.txt                               Oct 24, 2025     │
│  • data.csv                                   Oct 23, 2025     │
│  • report.pdf                                 Oct 22, 2025     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  🗄️ Datasets (3)                                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Email Dataset│  │ Legal Docs   │  │ Research     │         │
│  │              │  │              │  │              │         │
│  │ Contains...  │  │ Contains...  │  │ Contains...  │         │
│  │              │  │              │  │              │         │
│  │ 📄 125 files │  │ 📄 89 files  │  │ 📄 234 files │         │
│  │ 📅 Oct 20    │  │ 📅 Oct 18    │  │ 📅 Oct 15    │         │
│  │          [🗑️] │  │          [🗑️] │  │          [🗑️] │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Dataset Creator Page

```
┌─────────────────────────────────────────────────────────────────┐
│ [←] Create Dataset                          [💾 Save Dataset]   │
│     Select files and add annotations                            │
├─────────────────────────────────────────────────────────────────┤
│ ℹ️ 12 file(s) selected                                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📁 File Browser                  │  🤖 AI File Search          │
│                                   │                             │
│  🏠 Root > data > emails          │  Hi! I can help you find... │
│  ┌──────────────────────────────┐│  ┌────────────────────────┐│
│  │ ◁ .. (back)                  ││  │ 🤖 How can I help?     ││
│  │ ☐ 📁 inbox                   ││  │                        ││
│  │ ☐ 📁 sent                    ││  │ 👤 Find all txt files  ││
│  │ ☑ 📄 email1.txt      2.4 KB  ││  │                        ││
│  │ ☑ 📄 email2.txt      1.8 KB  ││  │ 🤖 I found 50 files... ││
│  │ ☐ 📄 email3.txt      3.1 KB  ││  │    • file1.txt         ││
│  │ ☐ 📄 report.pdf      156 KB  ││  │    • file2.txt         ││
│  │                              ││  │    [Add to Selection]  ││
│  │                              ││  │                        ││
│  └──────────────────────────────┘│  │ [Send message]         ││
│                                   │  └────────────────────────┘│
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📝 Dataset Information                                          │
│  ┌────────────────────────────────────────────────────────────┐│
│  │ Dataset Name *                                              ││
│  │ [Email Dataset Q4 2024_____________________________]        ││
│  │                                                             ││
│  │ Annotation / Description *                                  ││
│  │ ┌─────────────────────────────────────────────────────────┐││
│  │ │ This dataset contains all Q4 2024 email communications  │││
│  │ │ related to the legal department...                      │││
│  │ │                                                         │││
│  │ └─────────────────────────────────────────────────────────┘││
│  │ Add any metadata or information...                          ││
│  └────────────────────────────────────────────────────────────┘│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3. User Chat Page

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│                                                                  │
│                          💬                                      │
│                      User Chat                                   │
│                                                                  │
│              Chat functionality coming soon...                   │
│                                                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 User Flows

### Flow 1: Upload Files
```
1. User lands on Data Management page
2. Clicks upload area or drags file
3. File uploads with progress indicator
4. Success message appears
5. File appears in "Recently Uploaded" list
```

### Flow 2: Create Dataset (Manual Selection)
```
1. Click "Create Dataset" button
2. Navigate through file browser (like Finder)
3. Check boxes next to desired files
4. Fill in dataset name
5. Fill in annotation/description
6. Click "Save Dataset"
7. Redirected to Data Management page
8. New dataset appears in list
```

### Flow 3: Create Dataset (AI Search)
```
1. Click "Create Dataset" button
2. Type query in chatbot (e.g., "Find all PDFs about legal")
3. Chatbot searches and displays results
4. Click "Add to Selection" on results
5. Files automatically checked in browser
6. Fill in dataset information
7. Save dataset
```

### Flow 4: Delete Dataset
```
1. Click trash icon on dataset card
2. Confirmation dialog appears
3. User confirms deletion
4. Dataset removed from list
5. Success message appears
```

## 🎨 Color Scheme

- **Primary**: Blue (#0ea5e9) - Buttons, links, highlights
- **Background**: Light gray (#f9fafb) - Page background
- **Cards**: White (#ffffff) - Content containers
- **Text**: Dark gray (#111827) - Primary text
- **Text Secondary**: Medium gray (#6b7280) - Secondary text
- **Borders**: Light gray (#e5e7eb) - Dividers and borders
- **Success**: Green (#10b981) - Success messages
- **Error**: Red (#ef4444) - Error messages

## 📱 Responsive Design

The application is fully responsive:

- **Desktop (1024px+)**: Three-column layout in dataset creator
- **Tablet (768px-1023px)**: Two-column layout
- **Mobile (<768px)**: Single-column stacked layout

## 💡 UI Features

### Interactive Elements
- ✅ Hover effects on all clickable items
- ✅ Loading spinners during async operations
- ✅ Smooth transitions and animations
- ✅ Active state indicators
- ✅ Disabled state styling

### Feedback Mechanisms
- ✅ Toast notifications for success/error
- ✅ Inline validation messages
- ✅ Loading states with spinners
- ✅ Empty state illustrations
- ✅ Confirmation dialogs

### Accessibility
- ✅ Semantic HTML elements
- ✅ Proper form labels
- ✅ Keyboard navigation support
- ✅ Focus indicators
- ✅ Clear error messages

## 🖱️ Interactions

### File Browser
- **Click folder**: Navigate into directory
- **Click checkbox**: Toggle file selection
- **Click breadcrumb**: Jump to directory
- **Click back button**: Go up one level

### Search Chatbot
- **Type message**: Enter search query
- **Press Enter / Click Send**: Submit query
- **Click "Add to Selection"**: Bulk add files
- **Auto-scroll**: New messages appear at bottom

### Dataset Cards
- **Hover**: Elevation increases
- **Click trash icon**: Delete with confirmation
- **Shows**: Name, description, file count, date

## 🎯 Best Practices Implemented

1. **Clear visual hierarchy** - Important actions stand out
2. **Consistent spacing** - Using Tailwind's spacing scale
3. **Feedback on actions** - Users always know what's happening
4. **Error prevention** - Validation before submission
5. **Undo options** - Confirmation for destructive actions
6. **Familiar patterns** - Uses common UI conventions
7. **Loading states** - Never leave users wondering
8. **Empty states** - Helpful guidance when no data exists

## 🚀 Performance Optimizations

- React component memoization ready
- Efficient re-renders with proper state management
- Lazy loading preparation for large file lists
- Optimized images and icons
- Minimal bundle size with tree-shaking

---

**The UI is designed to be intuitive, modern, and delightful to use!** ✨

