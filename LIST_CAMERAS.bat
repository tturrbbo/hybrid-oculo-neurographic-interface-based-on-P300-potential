@echo off
cd /d "%~dp0"
call .venv\Scripts\activate
python -c "from eyegaze.utils.video import list_cameras; cams=list_cameras(); print('CAMERAS:'); [print(f'{i}: {name}') for i,name in enumerate(cams)]"
pause
