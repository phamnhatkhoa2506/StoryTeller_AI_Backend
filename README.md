## Environment: Python 3.12.10
    With pip:
    <pre>
        git clone <repo_url.git>
        cd StoryTeller_AI_Backend
        python -m venv venv
        source ./venv/bin/activate
        pip install -r requirements.txt
    </pre>

    With uv:
    <pre>
        git clone <repo_url.git>
        uv init StoryTeller_AI_Backend
        cd StoryTeller_AI_Backend
        uv venv .venv
        source .venv/bin/activate
        uv pip install -r requirements.txt
    <pre>

    On Windows, you can use:

    <pre>
        source .venv/Scripts/activate 
    </pre>


# Run code
    uvicorn src.app:app --reload

# Run test
    <pre>python -m tests.<test_file_name></pre>

    If you want to run single test case
    <pre>python -m unittest tests.test_<name>.Test<Name>.test_<method></method></pre>