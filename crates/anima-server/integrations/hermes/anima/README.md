# Anima memory provider for Hermes Agent

Makes [Anima](https://github.com/AnimaLab/Anima) the external memory of a
Hermes profile, so Hermes shares one memory with your other assistants.

## Install

```bash
anima-server connect hermes
```

This copies the plugin to `$HERMES_HOME/plugins/anima/` and sets
`memory.provider: anima` in `config.yaml`. To do it by hand, copy this folder
there and run `hermes memory setup`.

Only one external memory provider can be active in Hermes. If another one is
already selected, `connect` leaves it alone and registers Anima as an MCP
server instead.

## What it does

| Hermes hook | Behaviour |
|-------------|-----------|
| System prompt | A short static note on what Anima is and how to use its tools |
| First turn | Anima's session briefing: your profile, memories relevant to the working folder, stored skills |
| Every turn | Memories matching your message that the briefing did not already include |
| Built-in `memory` tool writes | Mirrored to Anima: adds are stored, edits and removals are applied to the matching memory |
| Tools | `memory_search`, `memory_add`, `memory_update`, `memory_delete`, `skill_list`, `skill_get`, `skill_save` |

Conversation turns are not uploaded. Anima stores facts, not transcripts: the
model saves what is worth keeping with `memory_add`, and Hermes's own notes are
mirrored.

Recall is bounded to three seconds. If the server is down, the turn proceeds
without recalled context.

## Configuration

First match wins:

1. `ANIMA_BASE_URL`, `ANIMA_API_KEY`, `ANIMA_NAMESPACE` (for example in Hermes's `.env`)
2. `$HERMES_HOME/anima.json` (written by `hermes memory setup`)
3. `~/.anima/client.json` (written by `anima-server connect --url ... --api-key ...`)
4. `http://127.0.0.1:3000`, namespace `default`

## What leaves Hermes

Your message text is sent to the Anima server as a search query on each turn,
and built-in memory writes are sent as memories. With the default address this
stays on your machine.
