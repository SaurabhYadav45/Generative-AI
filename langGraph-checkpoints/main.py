import os
from dotenv import load_dotenv
from graph import create_chat_graph
from langgraph.checkpoint.mongodb import MongoDBSaver
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
config = {"configurable": {"thread_id": "6"}}

def init():
    with MongoDBSaver.from_conn_string(MONGODB_URI) as checkpointer:
        graph_with_mongo = create_chat_graph(checkpointer=checkpointer)
        while True:
            user_input = input("> ")
            print("User input: ", user_input)
            # result = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
            for event in graph_with_mongo.stream({"messages": [{"role": "user","content":user_input}]}, config, stream_mode="values"):
                if "messages" in event:
                    event["messages"][-1].pretty_print()


init()

# python -u "c:\Users\saura\OneDrive\Documents\Desktop\Gen AI\langGraph-checkpoints\support.py"