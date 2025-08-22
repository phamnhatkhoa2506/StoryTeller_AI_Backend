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
    <pre>
        uvicorn src.app:app --reload
    </pre>
    

# Run test
    <pre>python -m tests.<test_file_name></pre>

    If you want to run single test case
    <pre>
        python -m unittest tests.test_<name>.Test<Name>.test_<method>
    </pre>

    Or:
    
    Use decorator to skip test method, example
    <pre>
        @unittest.skip("...")
        def test_method(self, ...):
            ...
    </pre>

# Create auth file for chroma 
    <pre>docker run --rm --entrypoint htpasswd httpd:2 -Bbn <username> <password> >> server.htpasswd</pre>

# Run docker compose
    <pre>docker compose up -d</pre>