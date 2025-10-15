import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def ensure_ffmpeg_hint() -> None:
    """On Windows, users often lack ffmpeg on PATH; provide a friendly hint if missing.

    We do not hard fail here because some environments bundle ffmpeg. Whisper backends
    typically require it; if it's missing, the underlying libraries will raise.
    """
    from shutil import which

    if which("ffmpeg") is None:
        print(
            "[hint] ffmpeg not found on PATH. Install from https://ffmpeg.org/ or winget: "
            "winget install Gyan.FFmpeg",
            file=sys.stderr,
        )


def transcribe_audio(
    audio_path: str,
    model_size: str = "small",
    device: str = "auto",
    compute_type: str = "int8",
    language: Optional[str] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Transcribe audio into text using faster-whisper.

    Returns overall transcript string and a list of segment dicts with timings.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is required. Install dependencies from requirements.txt"
        ) from exc

    # Device resolution
    if device == "auto":
        # faster-whisper selects GPU if available when device="auto"
        resolved_device = "auto"
    else:
        resolved_device = device

    model = WhisperModel(model_size, device=resolved_device, compute_type=compute_type)

    segments_iter, _info = model.transcribe(
        audio_path,
        language=language,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        beam_size=5,
        best_of=5,
        condition_on_previous_text=True,
    )

    segments: List[Dict[str, Any]] = []
    transcript_parts: List[str] = []
    for seg in segments_iter:
        seg_dict = {
            "start": float(seg.start),
            "end": float(seg.end),
            "text": seg.text.strip(),
        }
        segments.append(seg_dict)
        transcript_parts.append(seg_dict["text"]) 

    transcript = " ".join(transcript_parts).strip()
    return transcript, segments


def diarize_speakers(
    audio_path: str,
    hf_token: Optional[str] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Optional diarization using pyannote.audio.

    Returns list of regions with speaker labels: [{start, end, speaker}]. If unavailable, returns None.
    """
    token = hf_token or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        return None

    try:
        from pyannote.audio import Pipeline
    except Exception:
        return None

    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=token)
        diarization = pipeline(audio_path)
        regions: List[Dict[str, Any]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            regions.append(
                {
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "speaker": str(speaker),
                }
            )
        return regions
    except Exception:
        # Fail quietly; diarization is optional
        return None


def assign_speakers_to_segments(
    segments: List[Dict[str, Any]],
    regions: Optional[List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Assign speaker labels to ASR segments using simple time overlap with diarization regions.

    Returns new segments with optional 'speaker' and an estimated participants count.
    """
    if not regions:
        return segments, None

    labeled: List[Dict[str, Any]] = []
    speaker_set: set = set()
    for seg in segments:
        seg_mid = (seg["start"] + seg["end"]) / 2.0
        # Find the region that covers the midpoint of the segment
        speaker_label: Optional[str] = None
        for region in regions:
            if region["start"] <= seg_mid <= region["end"]:
                speaker_label = region["speaker"]
                break
        new_seg = dict(seg)
        if speaker_label:
            new_seg["speaker"] = speaker_label
            speaker_set.add(speaker_label)
        labeled.append(new_seg)

    participants = len(speaker_set) if speaker_set else None
    return labeled, participants


def chunk_by_speaker_turns(
    labeled_segments: List[Dict[str, Any]], 
    max_tokens: int = 1024
) -> List[str]:
    """Chunk transcript preserving speaker turn context."""
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0
    
    for seg in labeled_segments:
        speaker = seg.get("speaker", "Speaker")
        text = seg["text"]
        line = f"{speaker}: {text}"
        
        # Rough token estimate (words * 1.3)
        line_tokens = int(len(line.split()) * 1.3)
        
        if current_tokens + line_tokens > max_tokens and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_tokens = line_tokens
        else:
            current_chunk.append(line)
            current_tokens += line_tokens
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks


def chunk_text_semantic(text: str, max_tokens: int = 1024) -> List[str]:
    """Improved semantic chunking that preserves paragraph boundaries."""
    if not text:
        return []
    
    # Split by double newlines first (paragraphs), then sentences
    import re
    
    paragraphs = text.split("\n\n")
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    
    for para in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", para.strip())
        
        for s in sentences:
            words = len(s.split())
            estimated_tokens = int(words * 1.3)
            
            if current_len + estimated_tokens > max_tokens and current:
                chunks.append("\n".join(current))
                current = [s]
                current_len = estimated_tokens
            else:
                current.append(s)
                current_len += estimated_tokens
    
    if current:
        chunks.append("\n".join(current))
    
    return chunks


def summarize_transcript(
    transcript: str,
    labeled_segments: Optional[List[Dict[str, Any]]] = None,
    model_name: str = "knkarthick/MEETING_SUMMARY",
    max_input_tokens: int = 1024,
) -> str:
    """Enhanced summarization with better prompting and meeting-specific handling."""
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    except Exception as exc:
        raise RuntimeError(
            "transformers is required. Install dependencies from requirements.txt"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

    # Use speaker-aware chunking if available
    if labeled_segments and any("speaker" in s for s in labeled_segments):
        chunks = chunk_by_speaker_turns(labeled_segments, max_tokens=max_input_tokens)
    else:
        chunks = chunk_text_semantic(transcript, max_tokens=max_input_tokens)
    
    if not chunks:
        return ""

    # For short meetings, skip chunking
    if len(chunks) == 1:
        try:
            result = summarizer(
                chunks[0], 
                max_length=250, 
                min_length=80, 
                do_sample=False,
                num_beams=4
            )
            return result[0]["summary_text"].strip()
        except Exception:
            return "Summary generation failed for this transcript."

    # Map: Summarize each chunk
    partial_summaries: List[str] = []
    for i, ch in enumerate(chunks):
        try:
            # Add context about chunk position
            if len(chunks) > 1:
                position = "beginning" if i == 0 else "end" if i == len(chunks)-1 else "middle"
                prompt = f"Summarize this {position} section of a meeting:\n\n{ch}"
            else:
                prompt = ch
                
            result = summarizer(
                prompt, 
                max_length=200, 
                min_length=60, 
                do_sample=False,
                num_beams=4
            )
            partial_summaries.append(result[0]["summary_text"].strip())
        except Exception:
            # Skip failed chunks
            continue

    if not partial_summaries:
        return "Summary generation encountered errors."

    # Reduce: Combine partial summaries
    combined = "\n\n".join(partial_summaries)
    
    reduce_prompt = (
        "Create a concise meeting summary from these section summaries. "
        "Include: main discussion topics, key decisions, action items, and next steps. "
        "Maintain any speaker names or roles mentioned.\n\n"
        f"{combined}"
    )
    
    try:
        result = summarizer(
            reduce_prompt, 
            max_length=300, 
            min_length=100, 
            do_sample=False,
            num_beams=4
        )
        return result[0]["summary_text"].strip()
    except Exception:
        # Fallback: return combined partial summaries
        return combined


def format_minutes(
    meeting_datetime: Optional[str],
    participants_count: Optional[int],
    transcript: str,
    summary: str,
    speaker_segments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Produce final structured meeting minutes payload."""
    if meeting_datetime:
        dt_str = meeting_datetime
    else:
        now = dt.datetime.now()
        dt_str = now.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "meeting_datetime": dt_str,
        "participants_count": participants_count,
        "summary": summary,
        "transcript": transcript,
        "segments": speaker_segments,  # may include speaker labels if available
    }


def write_output(output: Optional[str], minutes: Dict[str, Any]) -> None:
    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(minutes, f, ensure_ascii=False, indent=2)
        print(f"Saved minutes to {out_path}")
    else:
        print(json.dumps(minutes, ensure_ascii=False, indent=2))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Meeting Minutes Generator (ASR + Summarization)")
    p.add_argument("audio", help="Path to audio/video file")
    p.add_argument("--datetime", dest="meeting_datetime", default=None, help="Meeting datetime, default now (YYYY-MM-DD HH:MM:SS)")
    p.add_argument("--model-size", default="small", help="faster-whisper model size (e.g., tiny, base, small, medium, large-v3)")
    p.add_argument("--device", default="auto", help="Device for ASR model: auto|cpu|cuda")
    p.add_argument("--compute-type", default="int8", help="Compute type for ASR: int8|int8_float16|float16|float32")
    p.add_argument("--language", default=None, help="Optional language code for ASR (e.g., en, fr)")
    p.add_argument("--hf-token", default=None, help="HuggingFace token for diarization (or set HUGGINGFACE_TOKEN env)")
    p.add_argument("--summary-model", default="knkarthick/MEETING_SUMMARY", help="Transformers model for summarization")
    p.add_argument("--max-chunk-tokens", type=int, default=1024, help="Chunk size for summarization input")
    p.add_argument("--output", default=None, help="Optional output JSON path")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    ensure_ffmpeg_hint()

    audio_path = args.audio
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    print("[1/4] Transcribing…", file=sys.stderr)
    transcript, segments = transcribe_audio(
        audio_path=audio_path,
        model_size=args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )

    print("[2/4] Diarizing speakers (optional)…", file=sys.stderr)
    regions = diarize_speakers(audio_path, hf_token=args.hf_token)
    labeled_segments, participants_count = assign_speakers_to_segments(segments, regions)

    # If we have labeled segments, prefer concatenating speaker-tagged transcript for better summaries
    if any("speaker" in s for s in labeled_segments):
        lines = []
        for s in labeled_segments:
            spk = s.get("speaker", "Speaker")
            lines.append(f"{spk}: {s['text']}")
        transcript_for_summary = "\n".join(lines)
    else:
        transcript_for_summary = transcript

    print("[3/4] Summarizing…", file=sys.stderr)
    summary = summarize_transcript(
        transcript_for_summary,
        labeled_segments=labeled_segments if any("speaker" in s for s in labeled_segments) else None,
        model_name=args.summary_model,
        max_input_tokens=args.max_chunk_tokens,
    )

    print("[4/4] Formatting output…", file=sys.stderr)
    minutes = format_minutes(
        meeting_datetime=args.meeting_datetime,
        participants_count=participants_count,
        transcript=transcript,
        summary=summary,
        speaker_segments=labeled_segments,
    )
    write_output(args.output, minutes)


if __name__ == "__main__":
    main()