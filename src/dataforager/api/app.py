from flask import Flask, request, jsonify
from flask_cors import CORS
from uuid import uuid4
import os
import logging
from dataforager.table_representation.openai_client import OpenAIClient
from dataforager.hyse.hypo_schema_search import hyse_search
from dataforager.actions.infer_action import infer_action, infer_mentioned_metadata_fields
from dataforager.actions.handle_action import handle_semantic_fields, handle_raw_fields
from dataforager.chat.handle_chat_history import append_user_query, append_system_response, get_user_queries, get_last_results, get_mentioned_fields


# Flask app configuration
app = Flask(__name__)
CORS(app)

# OpenAI client instantiation
openai_client = OpenAIClient()

# Logging configuration
logging.basicConfig(level=logging.INFO)

# In-memory storage for chat history
# TODO: Use server-side session to store chat histories
chat_history = {}

@app.route('/api/start_chat', methods=['POST'])
def start_new_chat_session():
    # Generate a unique session/thread ID
    thread_id = str(uuid4())
    # Initialize the chat history for this thread
    chat_history[thread_id] = []
    # Return the thread_id to the client
    return jsonify({"thread_id": thread_id})

@app.route('/api/get_chat_history', methods=['GET'])
def get_chat_history():
    # Extract thread_id from the request's query parameters
    thread_id = request.args.get('thread_id')
    
    # Check if thread_id is provided and valid
    if thread_id and thread_id in chat_history:
        # Return the chat history for the given thread_id
        return jsonify({
            "success": True,
            "thread_id": thread_id,
            "chat_history": chat_history[thread_id]
        })
    else:
        # Return an error message if thread_id is not provided or invalid
        return jsonify({
            "success": False,
            "error": "Invalid or missing thread_id"
        }), 400

@app.route('/api/update_chat_history', methods=['POST'])
def update_chat_history():
    data = request.get_json()
    thread_id, query = data.get('thread_id'), data.get('query')

    # Validate thread_id and query presence
    if not thread_id or thread_id not in chat_history:
        return jsonify({"success": False, "error": "Invalid or missing thread_id"}), 400
    if not query:
        return jsonify({"success": False, "error": "Missing query content"}), 400

    try:
        # Identify metadata fields
        # TODO: Make sure for user's initial query, mention_semantic_fields is always True
        semantic_fields_identified = infer_mentioned_metadata_fields(cur_query=query, semantic_metadata=True).get_true_fields()
        raw_fields_identified = infer_mentioned_metadata_fields(cur_query=query, semantic_metadata=False).get_true_fields()

        # Update chat history with additional metadata field information
        append_user_query(chat_history, thread_id, query, bool(semantic_fields_identified), bool(raw_fields_identified))
        logging.info(f"💬 Current chat history: {chat_history}")

        return jsonify({
                "success": True,
                "thread_id": thread_id,
                "chat_history": chat_history[thread_id]
        })
    except Exception as e:
        logging.error(f"Update chat history failed for thread: {thread_id}, Error: {e}")
        return jsonify({"error": "Update chat history failed due to an internal error"}), 500

@app.route('/api/hyse_search', methods=['POST'])
def initial_search():
    thread_id = request.get_json().get('thread_id')
    initial_query = request.json.get('query')

    if not initial_query or len(initial_query.strip()) == 0:
        logging.error("Empty query provided")
        return jsonify({"error": "No query provided"}), 400

    try:
        initial_results, _, _ = hyse_search(initial_query, search_space=None, num_schema=1, k=50)  # Keep top 50 results in initial search
        append_system_response(chat_history, thread_id, initial_results, refine_type="semantic")

        # Update the cached results
        chat_history[thread_id + '_cached_results'] = [result["table_name"] for result in initial_results]

        response_data = {
            "top_results": initial_results[:10],
            "complete_results": [result["table_name"] for result in initial_results],
        }

        logging.info(f"✅Search successful for query: {initial_query}")
        logging.info(f"💬 Current chat history: {chat_history}")

        return jsonify(response_data), 200
    except Exception as e:
        logging.error(f"Search failed for query: {initial_query}, Error: {e}")
        return jsonify({"error": "Search failed due to an internal error"}), 500

@app.route('/api/refine_search_space', methods=['POST'])
def refine_search_space():
    thread_id = request.get_json().get('thread_id')

    # Validate thread_id and query presence
    if not thread_id or thread_id not in chat_history:
        return jsonify({"success": False, "error": "Invalid or missing thread_id"}), 400
    
    try:
        # Get the cached results
        cached_results = chat_history.get(thread_id + "_cached_results")
        logging.info(f"📩Current cached results: {cached_results}")

        # Initialize defaults
        refined_results, inferred_semantic_fields, inferred_raw_fields = [], [], []

        # Get user current and previous queries
        cur_query, prev_query = get_user_queries(chat_history, thread_id)
        # Check if the current query mentions semantic / raw metadata fields
        mention_semantic_fields, mention_raw_fields = get_mentioned_fields(chat_history, thread_id)

        # Step 1: Determine action (reset / refine) based on query delta
        inferred_action = infer_action(cur_query=cur_query, prev_query=prev_query)
        logging.info(f"✅Inferred action for current query '{cur_query}' and previous query '{prev_query}': {inferred_action.model_dump()}")

        # Neither reset nor refine: error with LLM inference
        # TODO: implement retry mechanism
        if not inferred_action.reset and not inferred_action.refine:
            logging.error(f"Action inference failed: neither action was identified")
            return jsonify({"error": "Action inference failed"}), 500

        # Step 2.1: If RESET - restore last search results
        if inferred_action.reset:
            # Update the cached results to previous search results
            cached_results = get_last_results(chat_history, thread_id)
            chat_history[thread_id + "_cached_results"] = cached_results
            return jsonify({"success": True, "message": "Successfully reset to last results set"}), 200
        
        # Step 2.2: If REFINE
        # Step 3.1: Handle mentioned SEMANTIC metadata fields in user current query
        if mention_semantic_fields:
            # Identify mentioned semantic metadata fields
            inferred_semantic_fields = infer_mentioned_metadata_fields(cur_query=cur_query, semantic_metadata=True).get_true_fields()
            logging.info(f"✅Inferred mentioned semantic metadata fields for current query '{cur_query}': {inferred_semantic_fields}")

            # TODO: Filter semantic refined results based on cosine similarity scores for better precision?
            refined_results = handle_semantic_fields(chat_history, thread_id, search_space=cached_results)
            append_system_response(chat_history, thread_id, refined_results, refine_type="semantic")

        # Step 3.2: Handle mentioned RAW metadata fields in user current query
        if mention_raw_fields:
            # Identify mentioned raw metadata fields
            inferred_raw_fields = infer_mentioned_metadata_fields(cur_query=cur_query, semantic_metadata=False).get_true_fields()
            logging.info(f"✅Inferred mentioned raw metadata fields for current query '{cur_query}': {inferred_raw_fields}")
            
            refined_results = handle_raw_fields(cur_query, inferred_raw_fields, search_space=cached_results)
            append_system_response(chat_history, thread_id, refined_results, refine_type="raw")

        logging.info(f"💬Current chat history: {chat_history}")

        # Package the response with additional information from the second message onwards
        response_data = {
            "top_results": refined_results[:10],
            "complete_results": [result["table_name"] for result in refined_results],
            "inferred_action": inferred_action.get_true_fields(),
            "mention_semantic_fields": inferred_semantic_fields,
            "mention_raw_fields": inferred_raw_fields,
        }

        return jsonify(response_data), 200
        
    except Exception as e:
        logging.error(f"Search refinement failed, Error: {e}")
        return jsonify({"error": "Search refinement failed due to an internal error"}), 500

@app.route('/api/reset_search_space', methods=['POST'])
def reset_search_space():
    thread_id = request.get_json().get('thread_id')
    results = request.json.get('results')

    # Validate thread_id and results presence
    if not thread_id or thread_id not in chat_history:
        return jsonify({"success": False, "error": "Invalid or missing thread_id"}), 400

    if not results:
        return jsonify({"success": False, "error": "No results provided"}), 400

    try:
        # Update the cached results
        chat_history[thread_id + "_cached_results"] = results
        logging.info(f"💬Current chat history: {chat_history}")
        return jsonify({"success": True}), 200
    except Exception as e:
        logging.error(f"Search space reset failed for thread_id '{thread_id}', Error: {e}")
        return jsonify({"error": "Search space reset failed due to an internal error"}), 500


def main():
    """Console-script entry point: run the DataForager API server."""
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug)


if __name__ == '__main__':
    main()
