@echo off
setlocal

:: Step 1: Run the Docker Sphinx build
docker run --rm -v "D:\IISc\Courses\ProjectEmissions\ProjectPrograms\Github\EddyFlux":/app -w /app sphinx-docs make -C docs html

:: Step 2: Copy the built HTML files
xcopy /E /I /Y docs\build\html\* docs\

git init
:: Step 3: Git add
git add .

git remote add origin https://github.com/your-username/eddyflux-docs.git
git branch -M main

:: Step 4: Get user input for commit message
set /p commitmsg=Enter commit message: 

:: Step 5: Git commit and push
git commit -m "%commitmsg%"
git push

endlocal
