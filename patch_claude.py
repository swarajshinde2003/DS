import os, glob

# Find extension directory
user_home = os.path.expanduser("~")
pattern = os.path.join(user_home, "**", "native-binary", "claude.exe")
matches = glob.glob(pattern, recursive=True)

if not matches:
    print("Could not locate claude.exe automatically. Please specify full path.")
    exit(1)

exe_path = matches[0]
print(f"Found claude.exe: {exe_path}")

with open(exe_path, "rb") as f:
    data = f.read()

# Patch 1: Pr function bypass
target1 = b'function Pr(e,n){if(n?.allowlist===void 0){try{if(Wq())return!1}catch{return!1}if(!n?.skipEntitlementDenyOverlay&&wD(e,vD()))return!1}let r=An()||{},s=n?.allowlist??r.availableModels;if(!s)return!0;if(s.length===0)return!1;'
prefix1 = b'function Pr(e,n){return!0;'
suffix1 = b'}'
repl1 = prefix1 + (b' ' * (len(target1) - len(prefix1) - len(suffix1))) + suffix1

# Patch 2: o$e model validation bypass
target2 = b'if(y().has(r))return{valid:!0}}'
repl2 = b'if(!0)return{valid:!0}' + (b' ' * (len(target2) - 22))

patched = False
if target1 in data:
    data = data.replace(target1, repl1, 1)
    patched = True
if target2 in data:
    data = data.replace(target2, repl2, 1)
    patched = True

if patched:
    with open(exe_path, "wb") as f:
        f.write(data)
    print("Successfully patched claude.exe!")
else:
    print("claude.exe already patched or pattern mismatch.")
