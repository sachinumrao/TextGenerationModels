import streamlit as st


def load_model(model_name):
    # TODO: load model
    pass


if "model" not in st.session_state:
    model, tokenizer = load_model("LOTR")
    st.session_state["model"] = model
    st.session_state["tokenizer"] = tokenizer


def generate_text(model, tokenizer, text, temperature):
    # TODO: generate text
    pass


def main():
    st.title("Fan-Fiction Generator")
    st.text("Generate new lores with LOTR adapter")

    # add sidebar and put selectbox
    st.sidebar.title("Settings")
    model_name = st.sidebar.selectbox("Model", ["LOTR", "Potterverse", "Friends"])
    temperature = st.sidebar.slider("Temperature", 0.1, 1.0, 0.5)

    # text input
    text = st.text_input("Enter text", "Once upon a time")
    st.write("You entered:", text)

    # generate button
    if st.button("Generate"):
        st.write("Generating...")
        # TODO: implement generation logic
        st.write("Generated text:", text)

    # add slider to set temperature
    temp = st.slider("Temperature", 0.1, 1.0, 0.1)
    st.write("Temperature:", temp)

    # add dropdown with 4 options


if __name__ == "__main__":
    main()
