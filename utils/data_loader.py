def load_cornell_data(path="data/cornell", num_samples=2000):
    print("Loading Cornell Movie Dialogs...")
    lines = {}
    conversations = []

    with open(path + "/movie_lines.txt", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) >= 5:
                lines[parts[0]] = parts[-1].strip()

    with open(path + "/movie_conversations.txt", encoding="iso-8859-1") as f:
        for line in f:
            parts = line.split(" +++$+++ ")
            if len(parts) >= 4:
                ids = eval(parts[-1])
                for i in range(len(ids) - 1):
                    if ids[i] in lines and ids[i+1] in lines:
                        conversations.append((lines[ids[i]], lines[ids[i+1]]))

    conversations = [
        (a.lower(), b.lower())
        for a, b in conversations
        if 2 < len(a.split()) < 20 and 2 < len(b.split()) < 20
    ]

    print(f"Total pairs found: {len(conversations)}")
    return conversations[:num_samples]