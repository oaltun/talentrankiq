# CLI Protocol Versioning

The JSONL protocol between the GUI and `triq serve` uses semantic versioning.

## Version Format

`protocol_version` field in the `Request` envelope: `"1.2.0"` (major.minor.patch)

## Compatibility Rules

| Change Type | Version Bump | Example | GUI Behavior |
|-------------|--------------|---------|--------------|
| **Patch** | `1.2.0 → 1.2.1` | New optional field, new event type, new optional command param | ✅ Connect silently |
| **Minor** | `1.2.0 → 1.3.0` | New command, new optional response field, new enum variant (with `#[non_exhaustive]`) | ✅ Connect, maybe warn "new features available" |
| **Major** | `1.2.0 → 2.0.0` | Removed/renamed field, changed command signature, removed command, changed enum semantics | ❌ Reject with "Update GUI/CLI" dialog |

## GUI Handshake

On startup, the GUI sends a handshake request with its protocol version:

```json
{ "id": 0, "cmd": "handshake", "params": { "protocol_version": "1.2.0" } }
```

The server responds with its version:

```json
{ "id": 0, "ok": { "server_version": "1.3.0" } }
```

The GUI compares versions using semver:

- **Compatible**: `server_version` satisfies `^client_version` (same major, server >= client)
- **Incompatible**: Major version mismatch → show update dialog

## Server Implementation

```rust
fn handle_handshake(params: Value) -> Result<Value, String> {
    let client_version = params.get("protocol_version")
        .and_then(|v| v.as_str())
        .unwrap_or("0.0.0");
    let server_version = env!("CARGO_PKG_VERSION");
    
    if !semver_compatible(client_version, server_version) {
        return Err(format!(
            "Protocol version mismatch: client={}, server={}. Please update GUI or CLI.",
            client_version, server_version
        ));
    }
    
    ok_json(json!({ "server_version": server_version }))
}
```

## Adding Fields Safely

- **Always optional**: New fields must be `Option<T>` with `#[serde(default, skip_serializing_if = "Option::is_none")]`
- **Never remove**: Deprecate instead (keep field, ignore value)
- **Enums**: Add `#[non_exhaustive]` to allow new variants without breaking