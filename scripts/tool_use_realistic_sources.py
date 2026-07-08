from __future__ import annotations

from typing import Any

from tool_use_context_histories import (
    CONTEXT_PRESSURE_FINAL_REQUESTS,
    CONTEXT_PRESSURE_HISTORIES,
)


HOME_ASSISTANT_POINTERS = {
    "v1": {
        "dataset": "acon96/Home-Assistant-Requests (pattern pointer; fully rewritten)",
        "url": "https://huggingface.co/datasets/acon96/Home-Assistant-Requests",
        "license": "MIT",
    },
    "v2": {
        "dataset": "acon96/Home-Assistant-Requests-V2 (pattern pointer; fully rewritten)",
        "url": "https://huggingface.co/datasets/acon96/Home-Assistant-Requests-V2",
        "license": "MIT",
    },
}


HOME_ENTITY_IDS = (
    "light.kitchen_main",
    "light.kitchen_counter",
    "light.kitchen_sink",
    "light.kitchen_pendant",
    "light.living_room_main",
    "light.living_room_lamp",
    "light.living_room_cove",
    "light.bedroom_main",
    "light.bedroom_bedside_left",
    "light.bedroom_bedside_right",
    "light.study_main",
    "light.study_desk",
    "light.balcony",
    "light.balcony_string",
    "climate.bedroom_ac",
    "climate.bedroom_ac_guest",
    "climate.living_room_ac",
    "climate.study_ac",
    "lock.front_door",
    "lock.front_gate",
    "lock.balcony_door",
    "lock.store_room",
    "media_player.living_room_tv",
    "media_player.living_room_speaker",
    "media_player.bedroom_speaker",
    "media_player.kitchen_speaker",
    "cover.living_room_curtain",
    "cover.living_room_sheer",
    "cover.bedroom_curtain",
    "cover.bedroom_blackout",
    "cover.kitchen_blind",
    "cover.balcony_awning",
    "cover.study_blind",
    "media_player.balcony_speaker",
)


HOME_SYSTEM_PROMPT = """You may return at most one Home Assistant tool call:
- light.turn_off(entity_id: string)
- light.turn_on(entity_id: string, brightness: integer from 0 to 255)
- climate.set_temperature(entity_id: string, celsius: number)
- lock.lock(entity_id: string)
- media_player.play(entity_id: string, source: string)
- cover.close(entity_id: string)

Available entity IDs:
""" + "\n".join(f"- {entity_id}" for entity_id in HOME_ENTITY_IDS) + """

Choose one exact entity_id from the inventory. Similar names are distinct devices;
do not shorten an ID, translate it, or invent a room alias. Include every required
argument and no unrequested argument.

Return raw JSON with exactly these keys:
{"tool_call":"tool_name","arguments":{}}

Do not add Markdown fences, commentary, or any other keys."""


HOME_AUTOMATION_SPECS: dict[str, dict[str, Any]] = {
    "tool_use_003": {
        "request": "hey turn off the main kitchen light — leave the counter and sink ones alone",
        "tool_name": "light.turn_off",
        "arguments": {"entity_id": "light.kitchen_main"},
        "pointer": "v1",
    },
    "tool_use_004": {
        "request": "study desk light on please, brightness 90. not the main study light",
        "tool_name": "light.turn_on",
        "arguments": {"entity_id": "light.study_desk", "brightness": 90},
        "pointer": "v2",
    },
    "tool_use_007": {
        "request": "can you set our bedroom AC to 24 C? the normal bedroom, not guest room",
        "tool_name": "climate.set_temperature",
        "arguments": {"entity_id": "climate.bedroom_ac", "celsius": 24},
        "pointer": "v2",
    },
    "tool_use_008": {
        "request": "leaving now, lock the front gate. front door is already locked",
        "tool_name": "lock.lock",
        "arguments": {"entity_id": "lock.front_gate"},
        "pointer": "v1",
    },
    "tool_use_009": {
        "request": "play All India Radio News on the living room speaker pls, not on the TV",
        "tool_name": "media_player.play",
        "arguments": {
            "entity_id": "media_player.living_room_speaker",
            "source": "All India Radio News",
        },
        "pointer": "v2",
    },
    "tool_use_010": {
        "request": "close the bedroom blackout shade only. keep the bedroom curtain as it is",
        "tool_name": "cover.close",
        "arguments": {"entity_id": "cover.bedroom_blackout"},
        "pointer": "v1",
    },
}


def validate_realism_sources() -> None:
    if len(HOME_ENTITY_IDS) < 30 or len(set(HOME_ENTITY_IDS)) != len(HOME_ENTITY_IDS):
        raise ValueError("home inventory must contain at least 30 unique entity IDs")
    if len(HOME_AUTOMATION_SPECS) != 6:
        raise ValueError("expected exactly six home-automation replacements")
    if len(CONTEXT_PRESSURE_HISTORIES) != 4:
        raise ValueError("expected exactly four context-pressure histories")
    if set(CONTEXT_PRESSURE_FINAL_REQUESTS) != set(CONTEXT_PRESSURE_HISTORIES):
        raise ValueError("each context-pressure history needs one final request")
    call_counts: set[int] = set()
    history_word_counts: list[int] = []
    for history in CONTEXT_PRESSURE_HISTORIES.values():
        call_counts.add(
            sum(
                message["role"] == "assistant"
                and message["content"].startswith('{"tool_call"')
                for message in history
            )
        )
        history_word_counts.append(
            sum(len(message["content"].split()) for message in history)
        )
    if call_counts != {6, 7, 8, 9}:
        raise ValueError(f"context histories need 6/7/8/9 prior calls: {call_counts}")
    if min(history_word_counts) < 1500 or max(history_word_counts) > 3000:
        raise ValueError(f"context history word counts out of range: {history_word_counts}")
    if max(history_word_counts) - min(history_word_counts) < 300:
        raise ValueError(f"context history lengths are too uniform: {history_word_counts}")


validate_realism_sources()
