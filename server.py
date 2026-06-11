#!/usr/bin/env python3
"""
ChatInsights - WhatsApp Chat Analyzer Backend
============================================
Run with: python server.py

Server starts INSTANTLY. Sentiment model loads in the background.
"""

import re
import os
import logging
import threading
from collections import Counter, defaultdict
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import emoji

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Sentiment model — loads in background thread ────────────────────────────
_sentiment_pipeline = None
_model_loading = False
_model_ready = False

def _load_model_background():
    global _sentiment_pipeline, _model_loading, _model_ready
    _model_loading = True
    logger.info("Loading sentiment model in background...")
    try:
        from transformers import pipeline
        _sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True,
            max_length=512,
        )
        _model_ready = True
        logger.info("Sentiment model loaded successfully ✓")
    except Exception as e:
        logger.warning(f"Sentiment model failed to load: {e}. Using rule-based fallback.")
        _sentiment_pipeline = None
        _model_ready = True  # mark ready so analysis doesn't wait forever
    _model_loading = False

def get_sentiment_pipeline():
    return _sentiment_pipeline  # always returns immediately (None = use fallback)


# ─── WhatsApp Chat Parser ─────────────────────────────────────────────────────

ANDROID_PATTERN = re.compile(
    r'(\d{1,2}/\d{1,2}/\d{2,4}),?\s+'
    r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\s+-\s+'
    r'([^:]+?):\s+(.*)'
)
IOS_PATTERN = re.compile(
    r'\[(\d{1,2}/\d{1,2}/\d{2,4}),\s*(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s+'
    r'([^:]+?):\s+(.*)'
)

MEDIA_RE = re.compile(
    r'<Media omitted>|\u200eimage omitted|\u200evideo omitted|'
    r'\u200eaudio omitted|\u200edocument omitted|\u200esticker omitted|'
    r'\u200eGIF omitted|image omitted|video omitted|audio omitted',
    re.IGNORECASE
)
LINK_RE = re.compile(r'https?://\S+')

STOPWORDS = {
    'the','a','an','is','it','in','on','at','to','for','of','and','or','but',
    'not','be','was','are','were','been','being','have','has','had','do','does',
    'did','will','would','could','should','may','might','shall','can','i','you',
    'he','she','we','they','me','him','her','us','them','my','your','his','this',
    'that','these','those','so','if','as','by','with','from','up','about','into',
    'yes','no','ok','okay','yeah','yea','lol','haha','hahaha','its',"it's",
    "i'm","don't","doesn't","didn't","won't","can't",'just','like','even',
    'well','back','there','out','one','all','what','which','who','when','where',
    'how','why','get','got','also','than','then','now','only','too','very','much',
}

def _parse_datetime(date_str, time_str):
    for df in ['%d/%m/%Y','%m/%d/%Y','%d/%m/%y','%m/%d/%y']:
        for tf in ['%I:%M %p','%I:%M%p','%H:%M','%H:%M:%S','%I:%M:%S %p']:
            try:
                return datetime.strptime(f'{date_str} {time_str}', f'{df} {tf}')
            except ValueError:
                pass
    raise ValueError(f"Cannot parse: {date_str} {time_str}")

def extract_emojis(text):
    return [c for c in text if emoji.is_emoji(c)]

def parse_whatsapp_chat(text):
    messages = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = ANDROID_PATTERN.match(line) or IOS_PATTERN.match(line)
        if m:
            date_str, time_str, sender, body = m.group(1), m.group(2), m.group(3).strip(), m.group(4).strip()
            try:
                dt = _parse_datetime(date_str, time_str)
                messages.append({
                    'datetime': dt,
                    'sender': sender,
                    'body': body,
                    'is_media': bool(MEDIA_RE.search(body)),
                    'has_link': bool(LINK_RE.search(body)),
                    'emojis': extract_emojis(body),
                    'word_count': len(body.split()) if not MEDIA_RE.search(body) else 0,
                })
            except Exception:
                pass
        elif messages:
            messages[-1]['body'] += ' ' + line
    return messages


# ─── Sentiment helpers ────────────────────────────────────────────────────────

POSITIVE_WORDS = {
    'love','great','amazing','good','nice','happy','wonderful','excellent',
    'fantastic','awesome','best','thanks','thank','beautiful','perfect',
    'brilliant','super','glad','sweet','cool','fun','enjoy','enjoyed',
    'congratulations','congrats','pleased','blessed','grateful','haha','lol',
}
NEGATIVE_WORDS = {
    'bad','terrible','horrible','awful','hate','sad','sorry','wrong',
    'problem','issue','fail','failed','disappointed','upset','angry',
    'frustrated','worried','sick','miss','lost','broke','broken','hurt',
}

def _rule_based_sentiment(texts):
    scores = []
    for text in texts:
        words = set(text.lower().split())
        pos = len(words & POSITIVE_WORDS)
        neg = len(words & NEGATIVE_WORDS)
        total = pos + neg
        scores.append((pos - neg) / total if total else 0.0)
    return sum(scores) / len(scores) if scores else 0.0

def _score_to_mood(avg):
    if avg > 0.4:   return 'Very Positive'
    if avg > 0.1:   return 'Positive'
    if avg < -0.4:  return 'Very Negative'
    if avg < -0.1:  return 'Negative'
    return 'Neutral'

def _compute_sentiment(texts, sample=50):
    if not texts:
        return 0.0, 'Neutral'
    sampled = texts[:sample]
    pipe = get_sentiment_pipeline()
    if pipe is not None:
        try:
            results = pipe(sampled, batch_size=16, truncation=True, max_length=512)
            scores = [r['score'] if r['label']=='POSITIVE' else -r['score'] for r in results]
            avg = sum(scores) / len(scores)
            return avg, _score_to_mood(avg)
        except Exception as e:
            logger.warning(f"Transformer sentiment error: {e}")
    avg = _rule_based_sentiment(sampled)
    return avg, _score_to_mood(avg)

def _fmt_hour(h):
    if h == 0:    return '12 AM'
    if h < 12:    return f'{h} AM'
    if h == 12:   return '12 PM'
    return f'{h-12} PM'

def _compute_response_times(messages, senders):
    buckets = defaultdict(list)
    for i in range(1, len(messages)):
        prev, curr = messages[i-1], messages[i]
        if prev['sender'] != curr['sender']:
            delta = (curr['datetime'] - prev['datetime']).total_seconds() / 60
            if 0 < delta < 1440:
                buckets[curr['sender']].append(delta)
    result = []
    for s in senders:
        times = buckets.get(s, [])
        if times:
            result.append({'participant': s, 'avg_minutes': round(sum(times)/len(times), 1)})
    return sorted(result, key=lambda x: x['avg_minutes'])


# ─── Analysis Engine ──────────────────────────────────────────────────────────

def analyse_chat(messages):
    if not messages:
        raise ValueError("No messages found")

    senders = list(dict.fromkeys(m['sender'] for m in messages))
    total = len(messages)
    start_dt, end_dt = messages[0]['datetime'], messages[-1]['datetime']
    total_days = max((end_dt - start_dt).days, 1)

    all_words, all_emojis = [], []
    hourly, weekday, daily, monthly = Counter(), Counter(), Counter(), Counter()

    for m in messages:
        hourly[m['datetime'].hour] += 1
        weekday[m['datetime'].strftime('%A')] += 1
        daily[m['datetime'].strftime('%Y-%m-%d')] += 1
        monthly[m['datetime'].strftime('%b %Y')] += 1
        if not m['is_media']:
            words = [w.lower().strip('.,!?;:"\'-()[]{}') for w in m['body'].split()]
            all_words.extend([w for w in words if w and w not in STOPWORDS and len(w) > 1])
        all_emojis.extend(m['emojis'])

    most_active_day = max(daily, key=daily.get) if daily else ''
    best_hour = max(hourly, key=hourly.get) if hourly else 0

    # Per-participant
    participant_stats, participant_sentiments_list = [], []
    sender_texts = defaultdict(list)
    for m in messages:
        if not m['is_media'] and len(m['body']) > 10:
            sender_texts[m['sender']].append(m['body'])

    for sender in senders:
        p_msgs = [m for m in messages if m['sender'] == sender]
        p_words, p_emojis, p_hourly = [], [], Counter()
        for m in p_msgs:
            p_hourly[m['datetime'].hour] += 1
            if not m['is_media']:
                ws = [w.lower().strip('.,!?;:"\'-()[]{}') for w in m['body'].split()]
                p_words.extend([w for w in ws if w and len(w) > 1])
            p_emojis.extend(m['emojis'])

        emoji_counter = Counter(p_emojis)
        p_active_hour = _fmt_hour(max(p_hourly, key=p_hourly.get)) if p_hourly else 'N/A'
        score, mood = _compute_sentiment(sender_texts.get(sender, []))

        participant_stats.append({
            'name': sender,
            'message_count': len(p_msgs),
            'word_count': len(p_words),
            'avg_words_per_message': round(len(p_words) / max(len(p_msgs), 1), 1),
            'emoji_count': len(p_emojis),
            'media_count': sum(1 for m in p_msgs if m['is_media']),
            'link_count': sum(1 for m in p_msgs if m['has_link']),
            'message_percentage': round(len(p_msgs) / total * 100, 1),
            'most_active_hour': p_active_hour,
            'favorite_emoji': emoji_counter.most_common(1)[0][0] if emoji_counter else '',
            'sentiment_score': round(score, 3),
        })
        participant_sentiments_list.append({'name': sender, 'score': round(score, 3), 'mood': mood})

    overall_score, overall_mood = _compute_sentiment(
        [m['body'] for m in messages if not m['is_media'] and len(m['body']) > 10], sample=200
    )
    pos = sum(1 for s in participant_sentiments_list if s['score'] > 0.1)
    neg = sum(1 for s in participant_sentiments_list if s['score'] < -0.1)
    neu = len(participant_sentiments_list) - pos - neg
    tp  = max(len(participant_sentiments_list), 1)

    day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

    return {
        'overview': {
            'total_messages': total,
            'total_words': sum(m['word_count'] for m in messages),
            'total_characters': sum(len(m['body']) for m in messages),
            'total_participants': len(senders),
            'total_days': total_days,
            'start_date': start_dt.strftime('%d %b %Y'),
            'end_date': end_dt.strftime('%d %b %Y'),
            'avg_messages_per_day': round(total / total_days, 1),
            'most_active_day': most_active_day,
            'most_active_hour': _fmt_hour(best_hour),
            'total_media': sum(1 for m in messages if m['is_media']),
            'total_links': sum(1 for m in messages if m['has_link']),
            'total_emojis': len(all_emojis),
            'chat_type': 'group' if len(senders) > 2 else 'private',
        },
        'participants': sorted(participant_stats, key=lambda x: x['message_count'], reverse=True),
        'timeline': [{'date': d, 'count': c} for d, c in sorted(daily.items())],
        'hourly_activity': [{'hour': h, 'count': hourly.get(h, 0)} for h in range(24)],
        'weekday_activity': [{'day': d, 'count': weekday.get(d, 0)} for d in day_order],
        'top_words': [{'word': w, 'count': c} for w, c in Counter(all_words).most_common(50)],
        'top_emojis': [{'emoji': e, 'count': c} for e, c in Counter(all_emojis).most_common(20)],
        'sentiment': {
            'overall_score': round(overall_score, 3),
            'overall_mood': overall_mood,
            'positive_percent': round(pos / tp * 100, 1),
            'negative_percent': round(neg / tp * 100, 1),
            'neutral_percent': round(neu / tp * 100, 1),
            'participant_sentiments': participant_sentiments_list,
        },
        'media_stats': [
            {'type': 'Images', 'count': sum(1 for m in messages if m['is_media'])},
            {'type': 'Links',  'count': sum(1 for m in messages if m['has_link'])},
        ],
        'response_times': _compute_response_times(messages, senders),
        'monthly_activity': [
            {'month': mo, 'count': c}
            for mo, c in sorted(monthly.items(), key=lambda x: datetime.strptime(x[0], '%b %Y'))
        ],
    }


# ─── FastAPI App ──────────────────────────────────────────────────────────────

app = FastAPI(title='ChatInsights API', version='1.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])


@app.on_event('startup')
def startup_event():
    """Start model loading in background — server is immediately ready."""
    t = threading.Thread(target=_load_model_background, daemon=True)
    t.start()


@app.get('/health')
def health():
    return {
        'status': 'ok',
        'model_ready': _model_ready,
        'model_loading': _model_loading,
        'sentiment': 'transformer' if _sentiment_pipeline else ('loading' if _model_loading else 'rule-based'),
    }


@app.get('/info')
def info():
    return {
        'sentiment_model': 'distilbert-base-uncased-finetuned-sst-2-english' if _sentiment_pipeline else 'rule-based fallback',
        'model_loaded': _sentiment_pipeline is not None,
        'model_loading': _model_loading,
    }


@app.post('/analyze')
async def analyze(file: UploadFile = File(...)):
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail='Only .txt files are supported')

    content = await file.read()
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        text = content.decode('latin-1')

    logger.info(f'Received: {file.filename} ({len(text)} chars)')

    messages = parse_whatsapp_chat(text)
    if not messages:
        return {
            'success': False,
            'error': 'Could not parse any messages. Make sure this is a WhatsApp exported .txt file.',
        }

    logger.info(f'Parsed {len(messages)} messages from {len(set(m["sender"] for m in messages))} senders')

    try:
        data = analyse_chat(messages)
        return {'success': True, 'data': data}
    except Exception as e:
        logger.exception('Analysis error')
        return {'success': False, 'error': str(e)}


if __name__ == '__main__':
    print('=' * 55)
    print('  ChatInsights — WhatsApp Analyzer Backend')
    print('=' * 55)
    print('  Server : http://localhost:8080')
    print('  Status : http://localhost:8080/health')
    print('  Docs   : http://localhost:8080/docs')
    print()
    print('  Server starts instantly.')
    print('  Sentiment model loads in the background.')
    print('  Ctrl+C to stop.')
    print('=' * 55)

    port = int(os.environ.get('PORT', 8080))
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='info')