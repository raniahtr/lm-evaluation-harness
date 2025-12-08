#def doc_to_text(doc) -> str:
#   ctxs = "\n".join(doc["CONTEXTS"])
 #   return f"Abstract: {ctxs}\nQ: {doc['QUESTION']}\n" + "A: Provide the final answer [yes,no,maybe] enclosed in boxed{the_answer}."
    #return f"Q: {doc['QUESTION']}\n" + "A: Provide the final answer enclosed in boxed{the_answer}."


def doc_to_text(doc, use_schema: bool = True) -> str:
    ctxs = "\n".join(doc["CONTEXTS"])
    if use_schema:
        return (
            "You will read an abstract and a question. Answer the question based on the abstract.\n"
            f"Abstract: {ctxs}\n"
            f"Question: {doc['QUESTION']}\n"
        )
    else:
        return (
            "You will read an abstract and a question. Answer only with one of: yes, no, maybe.\n"
            "Output exactly boxed{yes} or boxed{no} or boxed{maybe}\n"
            f"Abstract: {ctxs}\n"
            f"Question: {doc['QUESTION']}\n"
        )
