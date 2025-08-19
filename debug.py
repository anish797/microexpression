import pandas as pd
import numpy as np
import os

def debug_casme_dataset(excel_path="../casme/annotations.xlsx", base_path="../casme/cropped/"):
    """Debug the CASME dataset to understand the data structure"""
    
    print("🔍 DEBUGGING CASME DATASET STRUCTURE")
    print("="*60)
    
    # Load dataset
    try:
        df = pd.read_excel(excel_path, header=None)
        df.columns = ['participant', 'video_name', 'start_frame', 'apex_frame', 
                      'end_frame', 'AUs', 'estimated_emotion', 'expression_type', 'self_reported_emotion']
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Extract emotion instances
    df['emotion_instance'] = df['video_name'].str.replace(r'_\d+$', '', regex=True)
    
    print(f"📊 BASIC DATASET INFO:")
    print(f"   Total samples: {len(df)}")
    print(f"   Participants: {df['participant'].nunique()}")
    print(f"   Unique video names: {df['video_name'].nunique()}")
    print(f"   Emotion classes: {df['emotion_instance'].nunique()}")
    
    # Check for potential data issues
    print(f"\n🔍 POTENTIAL DATA ISSUES:")
    
    # 1. Check for duplicate video names
    duplicate_videos = df['video_name'].value_counts()
    duplicates = duplicate_videos[duplicate_videos > 1]
    if len(duplicates) > 0:
        print(f"⚠️  DUPLICATE VIDEO NAMES FOUND:")
        for video, count in duplicates.head().items():
            print(f"   '{video}' appears {count} times")
            # Show participants for these duplicates
            dup_rows = df[df['video_name'] == video]
            participants = dup_rows['participant'].unique()
            print(f"     Participants: {participants}")
    else:
        print(f"✅ No duplicate video names found")
    
    # 2. Check subject distribution
    print(f"\n👥 SUBJECT DISTRIBUTION:")
    subject_counts = df['participant'].value_counts().sort_index()
    print(f"   Mean samples per subject: {subject_counts.mean():.1f}")
    print(f"   Std deviation: {subject_counts.std():.1f}")
    print(f"   Min samples: {subject_counts.min()}")
    print(f"   Max samples: {subject_counts.max()}")
    
    # Show extremely imbalanced subjects
    mean_samples = subject_counts.mean()
    high_subjects = subject_counts[subject_counts > mean_samples * 2]
    low_subjects = subject_counts[subject_counts < mean_samples * 0.5]
    
    if len(high_subjects) > 0:
        print(f"⚠️  HIGH-SAMPLE SUBJECTS (>{mean_samples*2:.0f} samples):")
        for subject, count in high_subjects.items():
            print(f"   Subject {subject}: {count} samples ({count/len(df)*100:.1f}%)")
    
    if len(low_subjects) > 0:
        print(f"⚠️  LOW-SAMPLE SUBJECTS (<{mean_samples*0.5:.0f} samples):")
        for subject, count in low_subjects.items():
            print(f"   Subject {subject}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # 3. Check emotion distribution
    print(f"\n😊 EMOTION DISTRIBUTION:")
    emotion_counts = df['emotion_instance'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count} samples ({count/len(df)*100:.1f}%)")
    
    # 4. Check for potential frame overlap issues
    print(f"\n🎬 FRAME RANGE ANALYSIS:")
    print(f"   Start frames - Min: {df['start_frame'].min()}, Max: {df['start_frame'].max()}")
    print(f"   Apex frames - Min: {df['apex_frame'].min()}, Max: {df['apex_frame'].max()}")
    print(f"   End frames - Min: {df['end_frame'].min()}, Max: {df['end_frame'].max()}")
    
    # Check for zero end frames
    zero_end = df[df['end_frame'] == 0]
    if len(zero_end) > 0:
        print(f"⚠️  {len(zero_end)} samples have end_frame = 0")
    
    # 5. Check sequence lengths
    df['sequence_length'] = df['end_frame'] - df['start_frame'] + 1
    print(f"\n📏 SEQUENCE LENGTHS:")
    print(f"   Mean length: {df['sequence_length'].mean():.1f} frames")
    print(f"   Min length: {df['sequence_length'].min()} frames")
    print(f"   Max length: {df['sequence_length'].max()} frames")
    print(f"   Std deviation: {df['sequence_length'].std():.1f} frames")
    
    # 6. Sample a few entries to check data structure
    print(f"\n📋 SAMPLE DATA ENTRIES:")
    sample_subjects = [1, 2, 15] if all(s in df['participant'].values for s in [1, 2, 15]) else df['participant'].unique()[:3]
    
    for subject in sample_subjects:
        subject_data = df[df['participant'] == subject].head(3)
        print(f"\n   Subject {subject} (showing first 3 samples):")
        for idx, row in subject_data.iterrows():
            print(f"     Video: {row['video_name']}")
            print(f"     Frames: {row['start_frame']}-{row['apex_frame']}-{row['end_frame']} (len: {row['sequence_length']})")
            print(f"     Emotion: {row['emotion_instance']}")
            
            # Check if the actual frame files exist
            participant_path = os.path.join(base_path, str(subject))
            video_path = os.path.join(participant_path, row['video_name'])
            if os.path.exists(video_path):
                frame_files = [f for f in os.listdir(video_path) if f.endswith('.jpg')]
                print(f"     Frame files available: {len(frame_files)}")
            else:
                print(f"     ⚠️  Video path not found: {video_path}")
            print()
    
    # 7. Check for potential data leakage patterns
    print(f"\n🔍 POTENTIAL DATA LEAKAGE ANALYSIS:")
    
    # Look for similar video names that might be from same original video
    video_prefixes = df['video_name'].str.extract(r'([^_]+_[^_]+)')[0].value_counts()
    repeated_prefixes = video_prefixes[video_prefixes > 1]
    
    if len(repeated_prefixes) > 0:
        print(f"⚠️  REPEATED VIDEO PREFIXES (potential splits from same video):")
        for prefix, count in repeated_prefixes.head(10).items():
            print(f"   '{prefix}*': {count} variations")
            # Show examples
            examples = df[df['video_name'].str.startswith(prefix)]['video_name'].unique()
            print(f"     Examples: {examples[:5]}")
    else:
        print(f"✅ No obvious video splitting patterns detected")
    
    # 8. Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    # Check if dataset is balanced enough for LOSO
    min_samples = subject_counts.min()
    max_samples = subject_counts.max()
    ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    if ratio > 10:
        print(f"⚠️  SEVERE IMBALANCE: Subject samples vary by {ratio:.1f}x")
        print(f"   Consider excluding subjects with <10 samples for reliable LOSO")
    elif ratio > 5:
        print(f"⚠️  MODERATE IMBALANCE: Subject samples vary by {ratio:.1f}x")
        print(f"   LOSO results may be unreliable for low-sample subjects")
    else:
        print(f"✅ Reasonable subject balance (ratio: {ratio:.1f}x)")
    
    # Recommend subjects for LOSO based on sample count
    good_subjects = subject_counts[subject_counts >= 15]  # At least 15 samples
    if len(good_subjects) > 0:
        print(f"\n✅ GOOD SUBJECTS FOR LOSO (≥15 samples):")
        for subject, count in good_subjects.sort_values(ascending=False).items():
            print(f"   Subject {subject}: {count} samples")
    
    poor_subjects = subject_counts[subject_counts < 10]  # Less than 10 samples
    if len(poor_subjects) > 0:
        print(f"\n❌ AVOID THESE SUBJECTS FOR LOSO (<10 samples):")
        for subject, count in poor_subjects.items():
            print(f"   Subject {subject}: {count} samples (too few for reliable testing)")
    
    return df

if __name__ == "__main__":
    df = debug_casme_dataset()
    
    print(f"\n🚀 Next steps:")
    print(f"1. Check if any data issues were found above")
    print(f"2. Use recommended subjects for LOSO testing")
    print(f"3. If 90%+ accuracy persists, suspect data leakage")