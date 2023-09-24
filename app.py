import os
import json
import torch
import pretty_midi
import streamlit as st

from GPT import *
from utils.utils import *

from loop_extraction.src.utils.remi import *
from loop_extraction.src.utils.constants import *
from loop_extraction.src.utils.bpe_encode import MusicTokenizer


st.set_page_config(
    layout="wide",
    page_title="Loop Generation",
)


@st.cache_resource()
def load_model(model_path, device_num):
    # load config
    with open("./config/config.json", "r") as f:
        config = json.load(f)

    # initialize model with GPU
    use_cuda = torch.cuda.is_available()
    device = torch.device(device_num if use_cuda else "cpu")

    # load tokenizer
    bpe_path = "./loop_extraction/tokenizer/tokenizer_meta.json"
    tokenizer = MusicTokenizer(bpe_path)

    pad_idx = tokenizer.encode([PAD_TOKEN])[0]
    vocab_size = tokenizer.bpe_vocab.get_vocab_size() + 1

    #### load model skeleton
    model = GPT(
        vocab_size,
        pad_idx,
        dim_model=config["dim_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        multiplier=config["multiplier"],
        dropout=config["dropout"],
        max_length=config["max_length"],
    )

    model = model.load_from_checkpoint(
        model_path,
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        dim_model=config["dim_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        multiplier=config["multiplier"],
        dropout=config["dropout"],
        max_length=config["max_length"],
        map_location=device,
    )

    return tokenizer, model


##########################
# MAIN BODY
##########################


def cs_body(tokenizer, model, device, max_length=1024):
    fs = 44100.0

    # special tokens
    bar_idx = tokenizer.encode([BAR_TOKEN])[0]
    pad_idx = tokenizer.encode([PAD_TOKEN])[0]
    eob_idx = tokenizer.encode([EOB_TOKEN])[0]

    # encoding start_token
    tokenizer.add_tokens([START_TOKEN])
    start_token = tokenizer.encode_meta(START_TOKEN)

    st.markdown(
        "<h1 style='text-align: center; padding: 6px;'><span style='color: rgb(96, 180, 255);'>Loop</span> Generation &#128526</h1>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align: center; font-size: 18px;'>Sangjun Han, Hyeongrae Ihm, Woohyung Lim from \
        <span style='font-weight: bold; color: rgb(165, 0, 52);'>LG AI Research</span></p>",
        unsafe_allow_html=True,
    )

    with st.form(key="my_form"):
        c1, c2, c3, c4, c5 = st.columns([0.07, 1, 0.07, 1, 0.07])

        with c2:
            st.markdown(
                "<h3 style='padding-bottom: 10px;'>For beginner mode</h3>",
                unsafe_allow_html=True,
            )

            inst = [f"{pretty_midi.program_to_instrument_name(i)}" for i in range(128)]
            inst.insert(0, "Drum")

            inst_options = st.multiselect(
                "Instruments",
                inst,
                max_selections=10,
                help="""You can choose multiple the instrument sets""",
            )

            num_loop = st.number_input(
                "The number of loops",
                value=1,
                min_value=1,
                max_value=8,
                help="""The repetition number for the loops""",
            )

            tempo = st.number_input(
                "Tempo",
                value=120,
                min_value=30,
                max_value=210,
                help="""Control tempo""",
            )

            top_k = st.slider(
                "Control Top_k",
                value=1,
                min_value=1,
                max_value=20,
                step=1,
                help="""The higher value, the more diverse musics""",
            )

            temperature = st.slider(
                "Control Temperature",
                value=0.5,
                min_value=0.5,
                max_value=1.5,
                step=0.1,
                help="""The higher value, the more diverse musics""",
            )

            submitted = st.form_submit_button(label="✨ Submit")

        with c4:
            st.markdown(
                "<h3 style='padding-bottom: 10px;'>For advanced mode</h3>",
                unsafe_allow_html=True,
            )

            st.write("Not mandatory, the controllability depends on the level of diversity")

            mean_pitch = st.number_input(
                "Mean Pitch",
                value=None,
                min_value=0,
                max_value=127,
                help="""Control mean pitch (absolute value [0 ~ 127])""",
            )

            mean_velocity = st.number_input(
                "Mean Velocity",
                value=None,
                min_value=1,
                max_value=127,
                help="""Control mean velocity (absolute value [1 ~ 127])""",
            )

            mean_duration = st.number_input(
                "Mean Duration",
                value=None,
                min_value=0,
                max_value=57,
                help="""Control mean duration (relative value [0 ~ 57])""",
            )

            chords = st.text_area(
                "Chords",
                placeholder="ex) C:maj, G:maj, A:min, E:min\n(only whose length is a multiple of 4 (max_length = 16))",
                help="""Allowed chord tones
                
                maj, min, dim, aug, dom7, maj7, min7
                """,
            )

        # generate samples
        if submitted:
            # instrument
            inst_token = [INSTRUMENT_KEY + "_" + option for option in inst_options]

            # mean_tempo
            tempo_idx = np.argmin(abs(DEFAULT_TEMPO_BINS - tempo))
            tempo_token = [TEMPO_KEY + "_" + str(tempo_idx)]

            # mean_pitch
            if mean_pitch is None:
                pitch_token = []
            else:
                pitch_token = [PITCH_KEY + "_" + str(mean_pitch)]

            # mean_velocity
            if mean_velocity is None:
                velocity_token = []
            else:
                velocity_idx = np.argmin(abs(DEFAULT_VELOCITY_BINS - mean_velocity))
                velocity_token = [VELOCITY_KEY + "_" + str(velocity_idx)]

            # mean_duration
            if mean_duration is None:
                duration_token = []
            else:
                duration_token = [DURATION_KEY + "_" + str(mean_duration)]

            # chords
            chord_token = [chord.strip() for chord in chords.split(",")]

            if len(chord_token) <= 1:
                chord_token = []
                bar_token = []
            else:
                try:
                    for chord in chord_token:
                        pitch, tone = chord.split(":")

                        if pitch not in PITCH_CLASSES or tone not in CHORD_TONE:
                            st.exception()

                    # bar_length
                    bar_token = ["Length_" + str(len(chord_token))]
                except:
                    st.error("Wrong chord format, the chord inputs will be ignored!")

                    # empty
                    chord_token = []
                    bar_token = []

            if len(chord_token) % 4 != 0 or len(chord_token) > 16:
                st.error("Only chords whose length is a multiple of 4 (<= 16), the chord inputs will be ignored!")

                # empty
                chord_token = []
                bar_token = []

            cond = []
            cond += sorted(tokenizer.encode(inst_token))
            cond += tokenizer.encode(pitch_token)
            cond += tokenizer.encode(tempo_token)
            cond += tokenizer.encode(velocity_token)
            cond += tokenizer.encode(duration_token)
            cond += tokenizer.encode_meta(chord_token)
            cond += tokenizer.encode_meta(bar_token)

            cond = start_token + cond + [bar_idx]
            cond = torch.tensor(cond, dtype=torch.long)
            print(cond)

            gen_loop = generate(
                cond,
                model,
                device,
                [bar_idx, eob_idx],
                temp=temperature,
                top_k=top_k,
                sample=True,
                max_length=max_length,
            )

            gen_loop = tokenizer.decode(gen_loop)

            repeat_loop = []
            for _ in range(num_loop):
                repeat_loop += gen_loop[:-1]
            repeat_loop += [EOB_TOKEN]

            loop_remi, end_time = remi2midi(repeat_loop)
            loop_fs = trim_tails(loop_remi, end_time, fs)

            with c2:
                st.audio(loop_fs, sample_rate=fs)


if __name__ == "__main__":
    model_path = "./model/GPT-Medium.ckpt"
    device_num = 0

    # load model
    tokenizer, model = load_model(model_path, device_num)
    # tokenizer, model = None, None

    # get front pages
    cs_body(tokenizer, model, device_num)
