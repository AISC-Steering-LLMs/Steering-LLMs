# Steering-LLMs

# In order to run code
Create Python virtual environment (place where python packages are installed) with ``python3 -m venv ~/camp``. Activate the virtual environment (which may have to be done each time you run code) with ``source ~/camp/bin/activate``. Then clone the repo and run ``python -m pip install -e Steering-LLMs`` from Python virtual env. Run the code with ``python3 Steering-LLMs/steerllm/main.py Steering-LLMs/data/inputs/prompts_good_evil_justice.xlsx``. This will create a folder with the experiment results in ``Steering-LLMs/data/output``.

# GitHub Basic Workflow
Adding things to the github involves a few steps. ``Monospace sections`` are intended to be run on the commandline/terminal. I am writing this tutorial using terminal commands, but there are GUI options for github as well.


1. Install GitHub
2. Clone the code ``git clone https://github.com/AISC-Steering-LLMs/Steering-LLMs`` (This may require you to configure your account locally. If this is needed, GitHub should link you to a guide to do so) (Need to change to a folder you are okay with before this. To put in the safety-camp folder inside Downloads folder, ``cd ~/Downloads`` ``mkdir safety-camp`` ``cd safety-camp``)
3. Create a branch to make your change on ``git branch my-branch``(this is stored separately, and can be viewed from the GitHub website. I have configured so that they will be deleted when code is merged)
4. Change your branch to the newly created one ``git checkout my-branch`` (the default branch is main. This is where your changes will eventually end up. You can go back to main with ``git checkout main``)
5. Add your file or make your changes within the folder. This may involve adding more prompts, adding notes, adding a run, or changing code
6. add your changes to git ``git add .`` (You can check that the new file/change has been added with ``git status``) (You may need to add a folder manually with ``git add Steering-LLMs/steerllm/data/outputs/2024-01-26_06-14-38/``)
7. commit your changes ``git commit -m "Added a new file"`` (-m indicates the message that you attach, so we can know what your goal is with the change) (At this point, your change is fully added to git)
8. push your changes to github server with ``git push --set-upstream origin my-branch`` (You should be able to view the changes on the GitHub website at this point. There is a dropdown at top left above the files/folders.)
9. Periodically run ``git pull origin main`` to download the latest changes from the server to your computer (It is best to run this before step 3, before creating a new branch) (run ``git config pull.rebase true`` to make sure this runs properly)
10. If GitHub says 'X commits behind main' on your branch, reach out to tech team. (You need to checkout your branch and run ``git rebase origin/main`` followed by ``git push --force-with-lease``. This can cause issues if done improperly)
11. Create a pull request (Go to GitHub website, Pull request page at the top. Click New pull request. Select your branch. Click Create pull request) (After this someone from the tech team can add your change to the main of GitHub)
12. ``git checkout main`` then ``git pull origin main`` to return to the main branch and download any updates, and be ready for any further changes. 
