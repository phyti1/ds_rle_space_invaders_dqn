@ECHO OFF
setlocal ENABLEDELAYEDEXPANSION

set end_from=.ipynb
set end_to=.py

REM pipenv install

for /r %%a in (*!end_from!) do (
	set b=%%a
	set b=!b:%end_from%=!!end_to!!!
	echo Converting %%a to !b!
	pipenv run jupytext --to py:percent "%%a"
)

pause