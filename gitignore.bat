:: To use drag and drop the files/folders you would like to ignore onto the script.
:: A .gitignore file will be created in the root directory of the files/folders that you drop on the script

:: Ex. I want to hide the file 'C:\myrepo\filetohide.txt'
:: When I drop this file onto this script a .gitignore 
:: file will be created at 'C:\myrepo\.gitignore'.

:: Originaly created by Ryan Esteves 1/19/2011

@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

for %%A in (%*) do (call :sub %%A)
goto end

:sub
set "dir=%~dp1"
set "out=%~1"
if not ("!out!"=="") (
	for %%i in (!out!) do ( 
		if exist %%~si\nul (
			set "out=!out!/"
		)
	)
	set "out=!out:%dir%=!"
	set "out=!out:\=/!"
	echo !out!>>.gitignore
	echo !out! added
)
goto :EOF
:end
PAUSE