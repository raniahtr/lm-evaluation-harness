def doc_to_text_old(doc) -> str:
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"({k}) {v}\n") for k, v in option_choices.items())
    return f"Q: {doc['sent1']}\n{answers}" + "A: Provide the final answer enclosed in boxed{the_answer}."

def doc_to_text_medqa_neutral(doc, use_schema: bool = True) -> str:
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"({k}) {v}\n" for k, v in option_choices.items()))
    if use_schema:
        return (
            f"Q: {doc['sent1']}\n"
            f"{answers}\n"
            "What is the correct answer?"
        )
    else:
        # Free-form prompt: explicit format instructions
        return (
            f"Q: {doc['sent1']}\n"
            f"{answers}\n"
            "Choose the correct answer from the options above.\n"
            "Output exactly boxed{A} or boxed{B} or boxed{C} or boxed{D}\n"
        )
    

