import json

def validate_openai_conversations(file_path):
    """
    Validates ShareGPT conversations for proper message order and role alternation.
    Returns a list of errors with detailed explanations.
    """
    try:
        with open(file_path, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        return [{"error": "File not found", "details": f"Path: {file_path}"}]
    except json.JSONDecodeError:
        return [{"error": "Invalid JSON format"}]

    errors = []

    for idx, conv_obj in enumerate(dataset):
        messages = conv_obj.get("conversations", [])
        if not messages:
            errors.append({
                "conversation_index": idx,
                "error": "Empty conversation",
                "details": "No messages found"
            })
            continue

        # Track expected next role after system messages
        expected_role = "user"
        system_found = False
        prev_role = None

        for msg_idx, message in enumerate(messages):
            role = message.get("role", "").lower()
            content = message.get("content", "")

            # Skip system messages but track their presence
            if role == "system":
                system_found = True
                continue

            # First non-system message must be from user
            if not system_found and msg_idx == 0 and role != "user":
                errors.append({
                    "conversation_index": idx,
                    "error": "Invalid first message",
                    "details": f"Expected 'user', got '{role}' at message {msg_idx}"
                })
                break

            # Check role alternation
            if role == prev_role and role in ("user", "assistant"):
                errors.append({
                    "conversation_index": idx,
                    "error": "Consecutive same-role messages",
                    "details": f"Message {msg_idx-1} and {msg_idx} both from '{role}'"
                })
                break

            if role not in ("user", "assistant", "system"):
                errors.append({
                    "conversation_index": idx,
                    "error": "Invalid role",
                    "details": f"Message {msg_idx}: Unknown role '{role}'"
                })
                break

            prev_role = role

        # Additional check for empty content
        if any(not msg.get("content", "").strip() for msg in messages):
            errors.append({
                "conversation_index": idx,
                "error": "Empty message content",
                "details": "Found message with empty or whitespace-only content"
            })

    return errors

def main():
    '''
    Validate the OpenAI conversations.
    '''
    # Usage example
    errors = validate_openai_conversations("esconv/train_openai.json")
    for error in errors:
        print(f"Conversation {error['conversation_index']}: {error['error']}")
        print(f"  Details: {error['details']}\n")
    if len(errors) == 0:
        print("No errors found in the dataset.")
    else:
        raise ValueError("Errors found in the dataset.")

if __name__ == "__main__":
    main()
