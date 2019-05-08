import bz2, base64

print(bz2.decompress(base64.b64decode('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')))
