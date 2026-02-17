# Fix Python PATH Issue

## Problem
Windows is trying to use the Microsoft Store Python stub instead of your actual Python installation.

## Solution Options

### Option 1: Use the Run Scripts (Easiest)
I've created two scripts for you:
- **run_app.bat** - Double-click this to run your app
- **run_app.ps1** - Right-click and "Run with PowerShell"

### Option 2: Fix PATH Permanently (Recommended)

1. **Open System Environment Variables:**
   - Press `Win + X` and select "System"
   - Click "Advanced system settings"
   - Click "Environment Variables"

2. **Edit PATH:**
   - Under "User variables", find and select "Path"
   - Click "Edit"
   - Find and **REMOVE** this entry:
     ```
     C:\Users\maddi\AppData\Local\Microsoft\WindowsApps
     ```
   - **ADD** this entry at the top (or near the top):
     ```
     C:\Users\maddi\AppData\Local\Programs\Python\Python311
     C:\Users\maddi\AppData\Local\Programs\Python\Python311\Scripts
     ```
   - Click OK on all dialogs

3. **Restart PowerShell/Terminal:**
   - Close and reopen your terminal
   - Test with: `python --version`

### Option 3: Disable Windows Store Python Stub

1. Open Windows Settings (Win + I)
2. Go to "Apps" → "Advanced app settings" → "App execution aliases"
3. Turn OFF the toggles for:
   - `python.exe`
   - `python3.exe`

## Verify It Works

After fixing PATH, test with:
```powershell
python --version
python -m streamlit --version
```

Then run your app:
```powershell
python -m streamlit run app.py
```

