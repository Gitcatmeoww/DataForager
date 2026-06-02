import logging

def append_user_query(chat_history, thread_id, text, mention_semantic_fields=None, mention_raw_fields=None):
    # Append user query with additional metadata field information to the chat history
    response = {
        "sender": "user",
        "text": text,
        "mention_semantic_fields": mention_semantic_fields if mention_semantic_fields is not None else False,
        "mention_raw_fields": mention_raw_fields if mention_raw_fields is not None else False
    }

    if thread_id in chat_history:
        chat_history[thread_id].append(response)
    
    # logging.info(f"💬User input query added to chat history for thread_id {thread_id}: {text}")

def append_system_response(chat_history, thread_id, text, refine_type):
    table_names = [result["table_name"] for result in text]
    
    # Append a system response to the chat history 
    response = {
        "sender": "system",
        "text": table_names,
        "refine_type": refine_type
    }

    if thread_id in chat_history:
        chat_history[thread_id].append(response)

    # logging.info(f"💬System response added to chat history for thread_id {thread_id}: {text}")

def get_user_queries(chat_history, thread_id):
    # Filter out only user messages
    user_messages = [message for message in chat_history[thread_id] if message["sender"] == "user"]
    
    # Get the last two user queries if available
    cur_query = user_messages[-1]["text"] if user_messages else None
    prev_query = user_messages[-2]["text"] if len(user_messages) > 1 else None

    return cur_query, prev_query

def get_mentioned_fields(chat_history, thread_id):
    # Get the last user query from the chat history
    last_user_message = next((msg for msg in reversed(chat_history[thread_id]) if msg["sender"] == "user"), None)

    mention_semantic_fields = last_user_message.get("mention_semantic_fields", False) if last_user_message else False
    mention_raw_fields = last_user_message.get("mention_raw_fields", False) if last_user_message else False

    return mention_semantic_fields, mention_raw_fields

def get_last_results(chat_history, thread_id):
    # Filter out last system response
    last_system_response = next((msg for msg in reversed(chat_history[thread_id]) if msg["sender"] == "system"), None)
    last_results = last_system_response["text"] if last_system_response else None

    return last_results