@ECHO OFF
setlocal ENABLEDELAYEDEXPANSION

set end_from=.py
set end_to=.ipynb

::pipenv install

for /r %%a in (*!end_from!) do (
	set b=%%a
	set b=!b:%end_from%=!!end_to!!!
	echo Converting %%a to !b!
	pipenv run jupytext --to notebook "%%a"
)

pause