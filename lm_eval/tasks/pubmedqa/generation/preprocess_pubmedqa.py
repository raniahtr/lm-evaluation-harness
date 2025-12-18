
def doc_to_text(doc, is_schema_mode_active: bool = True) -> str:
    ctxs = "\n".join(doc["CONTEXTS"])
    if is_schema_mode_active:
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
