def doc_to_text_old(doc) -> str:
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"({k}) {v}\n") for k, v in option_choices.items())
    return f"Q: {doc['sent1']}\n{answers}" + "A: Provide the final answer enclosed in boxed{the_answer}."

## 0.4988
def doc_to_text_medqa_neutral(doc) -> str:
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"({k}) {v}\n" for k, v in option_choices.items()))
    return (
        f"Q: {doc['sent1']}\n"
        f"{answers}\n"
        "What is the correct answer?"
    )
    
## 0.4407    
def doc_to_text_medqa_neutral_2(doc) -> str: 
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"({k}) {v}\n" for k, v in option_choices.items()))
    return (
        f"Q: {doc['sent1']}\n"
        f"{answers}\n"
        "What is the correct answer? Think carefully, then clearly state which option (A, B, C, or D) is correct."
    )
def doc_to_text_medqa_neutral_3(doc) -> str: 
    option_choices = {
        "A": doc["ending0"],
        "B": doc["ending1"],
        "C": doc["ending2"],
        "D": doc["ending3"],
    }
    answers = "".join((f"({k}) {v}\n" for k, v in option_choices.items()))
    return (
        f"Q: {doc['sent1']}\n"
        f"{answers}"
        "A:"
    )



def doc_to_target(doc) -> int:
    return doc["label"]
