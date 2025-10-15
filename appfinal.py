import streamlit as st
import tempfile
import os
from finalfinal import transcribe_audio, diarize_speakers, assign_speakers_to_segments, summarize_transcript, format_minutes
import json

st.set_page_config(
    page_title="Meeting Minute Generator",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    body {
        background-color: #0f172a;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    .title {
        text-align: center;
        color: #f1f5f9;
        font-size: 2.8rem;
        font-weight: 700;
        margin-top: 1rem;
    }
    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .stButton button {
        background-color: #2563eb;
        color: white;
        border-radius: 10px;
        font-size: 1.1rem;
        width: 100%;
        padding: 0.6rem;
        transition: 0.3s;
    }
    .stButton button:hover {
        background-color: #1e40af;
        transform: scale(1.02);
    }
    .card {
        background-color: #1e293b;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 3rem;
    }
    .upload-section {
        border: 2px dashed #3b82f6;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #1e293b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='title'>🎙️ Meeting Minute Generator</h1>", unsafe_allow_html=True)
st.markdown("""
<div class='subtitle'>
<b>AI-Powered Meeting Minutes Generator</b><br>
Upload your meeting audio and get instant transcription & summary!<br><br>

<b>✨ Features:</b><br>
🎙️ Upload audio files (MP3, WAV, M4A)<br>
🤖 Automatic transcription with Whisper AI<br>
👥 Optional speaker identification<br>
📝 Smart meeting summarization<br>
💾 Download results as JSON or TXT
</div>
""", unsafe_allow_html=True)

# Main upload section
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📁 Upload Your Audio File")

uploaded_file = st.file_uploader(
    "Choose an audio file", 
    type=["mp3", "wav", "m4a", "ogg", "flac"],
    help="Supported formats: MP3, WAV, M4A, OGG, FLAC"
)

if uploaded_file:
    st.success(f"✅ File uploaded: **{uploaded_file.name}** ({uploaded_file.size / 1024 / 1024:.2f} MB)")
    
    # Audio player
    st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
    
    st.markdown("---")
    
    # Configuration options
    st.markdown("### ⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_size = st.selectbox(
            "🔧 Whisper Model Size", 
            ["tiny", "base", "small", "medium", "large-v3"], 
            index=2,
            help="Larger models are more accurate but slower. 'small' is recommended for most use cases."
        )
        
        language = st.selectbox(
            "🌍 Language (optional)",
            ["Auto-detect", "English", "Spanish", "French", "German", "Italian", "Portuguese", "Dutch", "Russian", "Chinese", "Japanese", "Korean"],
            index=0,
            help="Leave as 'Auto-detect' to automatically identify the language"
        )
    
    with col2:
        summary_model = st.text_input(
            "📝 Summarization Model", 
            "knkarthick/MEETING_SUMMARY",
            help="HuggingFace model for generating summaries"
        )
        
        device = st.selectbox(
            "💻 Processing Device",
            ["auto", "cpu", "cuda"],
            index=0,
            help="'auto' will use GPU if available, otherwise CPU"
        )
    
    st.markdown("---")
    
    # Speaker diarization option
    enable_diarization = st.checkbox(
        "👥 Enable Speaker Diarization (Identify who spoke when)", 
        value=False,
        help="Requires a HuggingFace token. Get one from https://huggingface.co/settings/tokens"
    )
    
    hf_token = None
    if enable_diarization:
        hf_token = st.text_input(
            "🔑 HuggingFace Token:", 
            type="password",
            help="Your HuggingFace access token for speaker diarization"
        )
        if not hf_token:
            st.warning("⚠️ Please provide a HuggingFace token to enable speaker diarization")
    
    st.markdown("---")
    
    # Process button
    if st.button("🚀 Generate Transcript & Minutes", use_container_width=True, type="primary"):
        if enable_diarization and not hf_token:
            st.error("❌ Please provide a HuggingFace token for speaker diarization or disable the feature.")
        else:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_audio:
                temp_audio.write(uploaded_file.getvalue())
                audio_path = temp_audio.name

            try:
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Transcribe
                status_text.text("🎯 Step 1/4: Transcribing audio...")
                progress_bar.progress(25)
                
                lang_code = None if language == "Auto-detect" else language[:2].lower()
                
                transcript, segments = transcribe_audio(
                    audio_path, 
                    model_size=model_size,
                    device=device,
                    language=lang_code
                )
                
                # Step 2: Diarize (optional)
                progress_bar.progress(50)
                regions = None
                if enable_diarization and hf_token:
                    status_text.text("🎯 Step 2/4: Identifying speakers...")
                    regions = diarize_speakers(audio_path, hf_token=hf_token)
                    if regions:
                        st.info(f"✅ Speaker diarization completed!")
                    else:
                        st.warning("⚠️ Speaker diarization failed or returned no results")
                else:
                    status_text.text("🎯 Step 2/4: Skipping speaker diarization...")
                
                # Step 3: Assign speakers
                progress_bar.progress(75)
                status_text.text("🎯 Step 3/4: Processing segments...")
                labeled_segments, participants_count = assign_speakers_to_segments(segments, regions)

                # Prepare transcript for summary
                transcript_for_summary = transcript
                if any("speaker" in s for s in labeled_segments):
                    transcript_for_summary = "\n".join([
                        f"{s.get('speaker', 'Speaker')}: {s['text']}" 
                        for s in labeled_segments
                    ])

                # Step 4: Summarize
                status_text.text("🎯 Step 4/4: Generating summary...")
                summary = summarize_transcript(
                    transcript_for_summary,
                    labeled_segments=labeled_segments if any("speaker" in s for s in labeled_segments) else None,
                    model_name=summary_model,
                )

                # Format output
                minutes = format_minutes(
                    None, 
                    participants_count, 
                    transcript, 
                    summary, 
                    labeled_segments
                )

                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                
                st.balloons()
                
                # Clean up temp file
                os.unlink(audio_path)

                # Display results
                st.markdown("---")
                st.markdown("## 📊 Results")
                
                # Summary card
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### 📝 Meeting Summary")
                st.write(summary)
                if participants_count:
                    st.info(f"👥 Detected **{participants_count}** participants in the meeting")
                st.markdown("</div>", unsafe_allow_html=True)

                # Transcript card
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### 📋 Full Transcript")
                with st.expander("Click to view full transcript", expanded=False):
                    if any("speaker" in s for s in labeled_segments):
                        for seg in labeled_segments:
                            speaker = seg.get('speaker', 'Speaker')
                            timestamp = f"[{seg['start']:.1f}s - {seg['end']:.1f}s]"
                            st.markdown(f"**{speaker}** {timestamp}: {seg['text']}")
                    else:
                        st.text_area("Transcript", transcript, height=300)
                st.markdown("</div>", unsafe_allow_html=True)

                # Download section
                st.markdown("---")
                st.markdown("### 💾 Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.download_button(
                        label="📥 Download Full Minutes (JSON)",
                        data=json.dumps(minutes, indent=2),
                        file_name=f"meeting_minutes_{uploaded_file.name.split('.')[0]}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                
                with col2:
                    st.download_button(
                        label="📥 Download Summary (TXT)",
                        data=summary,
                        file_name=f"meeting_summary_{uploaded_file.name.split('.')[0]}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                
                with col3:
                    st.download_button(
                        label="📥 Download Transcript (TXT)",
                        data=transcript,
                        file_name=f"transcript_{uploaded_file.name.split('.')[0]}.txt",
                        mime="text/plain",
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.exception(e)
                # Clean up temp file on error
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

else:
    st.markdown("""
    <div class='upload-section'>
        <h3>👆 Click above to upload your audio file</h3>
        <p>Drag and drop or browse for files</p>
        <p style='color: #64748b; font-size: 0.9rem;'>
            Supported formats: MP3, WAV, M4A, OGG, FLAC<br>
            Maximum file size: 200MB
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<p class='footer'>© 2025 Meeting Minute Generator | Powered by Whisper AI & Transformers</p>", unsafe_allow_html=True)

# Sidebar with instructions
with st.sidebar:
    st.markdown("## 📖 How to Use")
    st.markdown("""
    1. **Upload** your audio file
    2. **Configure** model settings (optional)
    3. **Enable** speaker diarization (optional)
    4. **Click** Generate button
    5. **Download** your results!
    
    ---
    
    ### 💡 Tips
    - Use **'small'** model for faster processing
    - Use **'medium'** or **'large-v3'** for better accuracy
    - Enable **speaker diarization** to identify who spoke when
    - Longer files may take several minutes to process
    
    ---
    
    ### 🔧 Requirements
    - Valid audio file (clear recording recommended)
    - HuggingFace token (only for speaker diarization)
    - Stable internet connection
    """)
    
    st.markdown("---")
    st.markdown("### 🆘 Need Help?")
    st.markdown("""
    - [HuggingFace Token](https://huggingface.co/settings/tokens)
    - [Whisper Models](https://github.com/openai/whisper)
    - [PyAnnote Diarization](https://github.com/pyannote/pyannote-audio)
    """)