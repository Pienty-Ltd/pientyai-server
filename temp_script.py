#!/usr/bin/env python3
import re

# Read the file
with open('app/core/services/document_analysis_service.py', 'r') as f:
    content = f.read()

# First replacement - update_analysis_record
pattern1 = r'(# Update the analysis record with the diff changes\s+analysis\.diff_changes = analysis_data\.get\("diff_changes", ""\)\s+\s+# Keep legacy fields for backward compatibility\s+analysis\.analysis = analysis_data\.get\("analysis", ""\)\s+analysis\.key_points = analysis_data\.get\("key_points", \[\]\)\s+analysis\.conflicts = analysis_data\.get\("conflicts", \[\]\)\s+analysis\.recommendations = analysis_data\.get\("recommendations", \[\]\))'
replacement1 = '# Update the analysis record with only diff changes\n                analysis.diff_changes = analysis_data.get("diff_changes", "")'

# Second replacement - update_analysis_record_by_fp
pattern2 = r'(# Update the analysis record with the results\s+analysis\.analysis = analysis_data\.get\(\s+"analysis"\) or ""  # Ensure analysis is never None\s+analysis\.key_points = analysis_data\.get\("key_points", \[\]\)\s+analysis\.conflicts = analysis_data\.get\("conflicts", \[\]\)\s+analysis\.recommendations = analysis_data\.get\(\s+"recommendations", \[\]\))'
replacement2 = '# Update the analysis record with only diff changes\n                analysis.diff_changes = analysis_data.get("diff_changes", "")'

# Perform the replacements
modified_content = re.sub(pattern1, replacement1, content, flags=re.MULTILINE)
modified_content = re.sub(pattern2, replacement2, modified_content, flags=re.MULTILINE)

# Write the modified content back to the file
with open('app/core/services/document_analysis_service.py', 'w') as f:
    f.write(modified_content)

print("Replacements completed")