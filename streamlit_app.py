"""
Streamlit UI for Azerbaijani NLP Tasks
Interactive interface for visualizing and testing NLP models
"""
import os, json
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import src.extra_weighted as extra_weighted
import src.tokenization as tokenization
import src.sentence_segmentation as sentence_segmentation
import src.spell_checking as spell_checking

# Set page config
st.set_page_config(
    page_title="Azerbaijani NLP Tool",
    page_icon="🇦🇿",
    layout="wide"
)

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .task-header {
        font-size: 1.8rem;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def load_models(output_dir='./outputs'):
    """Load all trained models"""
    models = {}
    
    try:
        # Load tokenizer
        vocab_path = os.path.join(output_dir, 'vocabulary.pkl')
        if os.path.exists(vocab_path):
            tokenizer = tokenization.Tokenizer()
            tokenizer.load_vocabulary(vocab_path)
            models['tokenizer'] = tokenizer
        
        # Load BPE
        bpe_path = os.path.join(output_dir, 'bpe_model.pkl')
        if os.path.exists(bpe_path):
            bpe = bpe.BPETokenizer()
            bpe.load_model(bpe_path)
            models['bpe'] = bpe
        
        # Load sentence segmenter
        seg_path = os.path.join(output_dir, 'sentence_segmenter.pkl')
        if os.path.exists(seg_path):
            segmenter = sentence_segmentation.SentenceSegmenter()
            segmenter.load_model(seg_path)
            models['segmenter'] = segmenter
        
        # Load spell checker
        spell_path = os.path.join(output_dir, 'spell_checker.pkl')
        if os.path.exists(spell_path):
            spell_checker = spell_checking.SpellChecker()
            spell_checker.load_model(spell_path)
            models['spell_checker'] = spell_checker
        
        # Load weighted spell checker
        weighted_path = os.path.join(output_dir, 'weighted_spell_checker.pkl')
        if os.path.exists(weighted_path):
            weighted_checker = extra_weighted.WeightedSpellChecker()
            weighted_checker.load_model(weighted_path)
            models['weighted_checker'] = weighted_checker
        
        # Load summary
        summary_path = os.path.join(output_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                models['summary'] = json.load(f)
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
    
    return models


def show_overview(models):
    """Show overview of all tasks"""
    st.markdown('<h1 class="main-header">🇦🇿 Azerbaijani NLP Processing Tool</h1>', unsafe_allow_html=True)
    
    if 'summary' in models:
        summary = models['summary']
        
        st.markdown("### 📊 Dataset Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Files", summary['data_statistics']['total_files'])
        with col2:
            st.metric("Total Pages", summary['data_statistics']['total_pages'])
        with col3:
            st.metric("Total Characters", f"{summary['data_statistics']['total_characters']:,}")
        with col4:
            st.metric("Total Lines", summary['data_statistics']['total_lines'])
        
        st.markdown("### 📈 Task Results Summary")
        
        # Create results table
        results_data = []
        
        if 'task1' in summary:
            results_data.append({
                'Task': 'Task 1: Tokenization',
                'Key Metric': 'Type-Token Ratio',
                'Value': f"{summary['task1']['type_token_ratio']:.4f}",
                'Details': f"{summary['task1']['total_types']:,} types, {summary['task1']['total_tokens']:,} tokens"
            })
        
        if 'task2' in summary:
            results_data.append({
                'Task': 'Task 2: Heaps\' Law',
                'Key Metric': 'β (beta)',
                'Value': f"{summary['task2']['beta']:.4f}",
                'Details': f"k={summary['task2']['k']:.4f}, R²={summary['task2']['r_squared']:.4f}"
            })
        
        if 'task3' in summary:
            results_data.append({
                'Task': 'Task 3: BPE',
                'Key Metric': 'Vocabulary Size',
                'Value': f"{summary['task3']['vocabulary_size']:,}",
                'Details': f"{summary['task3']['num_merges']} merges performed"
            })
        
        if 'task4' in summary:
            results_data.append({
                'Task': 'Task 4: Sentence Segmentation',
                'Key Metric': 'Total Sentences',
                'Value': f"{summary['task4']['total_sentences']:,}",
                'Details': f"Avg length: {summary['task4']['avg_sentence_length']:.2f} words"
            })
        
        if 'task5' in summary:
            results_data.append({
                'Task': 'Task 5: Spell Checker',
                'Key Metric': 'Vocabulary Size',
                'Value': f"{summary['task5']['vocabulary_size']:,}",
                'Details': 'Levenshtein distance based'
            })
        
        if 'extra_task' in summary:
            results_data.append({
                'Task': 'Extra: Weighted Edit Distance',
                'Key Metric': 'Confusion Patterns',
                'Value': f"{summary['extra_task']['confusion_patterns']}",
                'Details': 'Character confusion matrix learned'
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    else:
        st.warning("No summary data found. Please run the main pipeline first.")
        st.code("python main.py --sample --output ./outputs")


def show_task1(models):
    """Show Task 1: Tokenization"""
    st.markdown('<h2 class="task-header">📝 Task 1: Tokenization & Frequency Analysis</h2>', unsafe_allow_html=True)
    
    if 'tokenizer' not in models:
        st.error("Tokenizer model not found. Please run Task 1 first.")
        return
    
    tokenizer = models['tokenizer']
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Tokens", f"{len(tokenizer.tokens):,}")
    with col2:
        st.metric("Unique Types", f"{len(set(tokenizer.tokens)):,}")
    with col3:
        ratio = len(set(tokenizer.tokens)) / len(tokenizer.tokens) if tokenizer.tokens else 0
        st.metric("Type-Token Ratio", f"{ratio:.4f}")
    
    # Top tokens
    st.markdown("### Most Frequent Tokens")
    top_tokens = tokenizer.token_freq.most_common(20)
    df = pd.DataFrame(top_tokens, columns=['Token', 'Frequency'])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        tokens = [t[0] for t in top_tokens[:10]]
        freqs = [t[1] for t in top_tokens[:10]]
        ax.barh(tokens, freqs)
        ax.set_xlabel('Frequency')
        ax.set_title('Top 10 Most Frequent Tokens')
        ax.invert_yaxis()
        st.pyplot(fig)
    
    # Test tokenization
    st.markdown("### Test Tokenization")
    test_text = st.text_area("Enter Azerbaijani text to tokenize:", 
                             value="Azərbaycan Respublikası Cənubi Qafqazda yerləşir.",
                             height=100)
    
    if st.button("Tokenize", key="tokenize_btn"):
        tokens = tokenizer.tokenize(test_text)
        st.write(f"**Tokens ({len(tokens)}):**")
        st.write(tokens)


def show_task2(models):
    """Show Task 2: Heaps' Law"""
    st.markdown('<h2 class="task-header">📈 Task 2: Heaps\' Law Analysis</h2>', unsafe_allow_html=True)
    
    # Load Heaps' Law plot
    plot_path = './outputs/heaps_law_plot.png'
    params_path = './outputs/heaps_law_params.json'
    
    if os.path.exists(params_path):
        with open(params_path, 'r') as f:
            params = json.load(f)
        
        st.markdown("### Heaps' Law Formula")
        st.latex(r"V(n) = k \cdot n^{\beta}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("k (constant)", f"{params['k']:.4f}")
        with col2:
            st.metric("β (beta)", f"{params['beta']:.4f}")
        with col3:
            st.metric("R² (goodness of fit)", f"{params['r_squared']:.4f}")
        
        st.markdown("### Interpretation")
        beta = params['beta']
        if 0.4 <= beta <= 0.6:
            st.success(f"β = {beta:.4f} indicates **typical vocabulary growth** for natural language.")
        else:
            st.info(f"β = {beta:.4f} indicates **atypical vocabulary growth** (typical range: 0.4-0.6).")
        
        if os.path.exists(plot_path):
            st.markdown("### Vocabulary Growth Visualization")
            st.image(plot_path, use_container_width=True)
    else:
        st.error("Heaps' Law results not found. Please run Task 2 first.")


def show_task3(models):
    """Show Task 3: BPE"""
    st.markdown('<h2 class="task-header">🔤 Task 3: Byte Pair Encoding (BPE)</h2>', unsafe_allow_html=True)
    
    if 'bpe' not in models:
        st.error("BPE model not found. Please run Task 3 first.")
        return
    
    bpe = models['bpe']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vocabulary Size", f"{len(bpe.token_to_id):,}")
    with col2:
        st.metric("Number of Merges", f"{len(bpe.merges):,}")
    
    # Show sample merges
    st.markdown("### Sample BPE Merges")
    sample_merges = bpe.merges[:20]
    merge_df = pd.DataFrame([
        {'Step': i+1, 'Merge': f"'{a}' + '{b}' → '{a}{b}'"}
        for i, (a, b) in enumerate(sample_merges)
    ])
    st.dataframe(merge_df, use_container_width=True, hide_index=True)
    
    # Test BPE encoding
    st.markdown("### Test BPE Encoding")
    test_text = st.text_input("Enter text to encode:",
                              value="Azərbaycan")
    
    if st.button("Encode with BPE", key="bpe_encode_btn"):
        encoded = bpe.encode(test_text)
        decoded = bpe.decode(encoded)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original:**")
            st.code(test_text)
            st.write("**Encoded tokens:**")
            st.code(' | '.join(encoded))
        with col2:
            st.write("**Decoded:**")
            st.code(decoded)
            st.write("**Token count:**")
            st.info(f"{len(encoded)} tokens")


def show_task4(models):
    """Show Task 4: Sentence Segmentation"""
    st.markdown('<h2 class="task-header">✂️ Task 4: Sentence Segmentation</h2>', unsafe_allow_html=True)
    
    if 'segmenter' not in models:
        st.error("Sentence segmenter not found. Please run Task 4 first.")
        return
    
    segmenter = models['segmenter']
    
    # Load sentences
    sentences_path = './outputs/sentences.json'
    if os.path.exists(sentences_path):
        with open(sentences_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            sentences = data['sentences']
        
        st.metric("Total Sentences Found", f"{len(sentences):,}")
        
        # Show sample sentences
        st.markdown("### Sample Sentences")
        for i, sent in enumerate(sentences[:5], 1):
            st.write(f"**{i}.** {sent}")
    
    # Test segmentation
    st.markdown("### Test Sentence Segmentation")
    test_text = st.text_area(
        "Enter text to segment:",
        value="Azərbaycan Respublikası Cənubi Qafqazda yerləşir. Ölkənin paytaxtı Bakı şəhəridir. Prof. Əliyev bu mövzuda tədqiqat aparır!",
        height=150
    )
    
    if st.button("Segment Sentences", key="segment_btn"):
        sentences = segmenter.segment_advanced(test_text)
        
        st.write(f"**Found {len(sentences)} sentence(s):**")
        for i, sent in enumerate(sentences, 1):
            st.write(f"**{i}.** {sent}")


def show_task5(models):
    """Show Task 5: Spell Checker"""
    st.markdown('<h2 class="task-header">✍️ Task 5: Spell Checker (Levenshtein Distance)</h2>', unsafe_allow_html=True)
    
    if 'spell_checker' not in models:
        st.error("Spell checker not found. Please run Task 5 first.")
        return
    
    spell_checker = models['spell_checker']
    
    st.metric("Vocabulary Size", f"{len(spell_checker.vocabulary):,}")
    
    # Test spell checking
    st.markdown("### Test Spell Checker")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        test_word = st.text_input("Enter a word to check:",
                                  value="azrbaycan")
    
    with col2:
        max_distance = st.slider("Max edit distance:", 1, 3, 2)
    
    if st.button("Check Spelling", key="spell_check_btn"):
        suggestions = spell_checker.correct_word(test_word, max_distance=max_distance, top_n=5)
        
        if suggestions and suggestions[0][1] == 0:
            st.success(f"✓ '{test_word}' is spelled correctly!")
        elif suggestions:
            st.warning(f"'{test_word}' might be misspelled. Suggestions:")
            
            sugg_df = pd.DataFrame([
                {
                    'Suggestion': sugg,
                    'Edit Distance': dist,
                    'Frequency': freq
                }
                for sugg, dist, freq in suggestions
            ])
            st.dataframe(sugg_df, use_container_width=True, hide_index=True)
        else:
            st.error("No suggestions found. The word is too different from vocabulary words.")
    
    # Batch correction
    st.markdown("### Batch Text Correction")
    batch_text = st.text_area("Enter text with potential errors:",
                              value="azrbaycan respublikası cənubi qafqazda yerləşir",
                              height=100)
    
    if st.button("Correct Text", key="correct_text_btn"):
        corrected = spell_checker.correct_text(batch_text, max_distance=2)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Original:**")
            st.code(batch_text)
        with col2:
            st.write("**Corrected:**")
            st.code(corrected)


def show_extra_task(models):
    """Show Extra Task: Weighted Edit Distance"""
    st.markdown('<h2 class="task-header">⚖️ Extra Task: Weighted Edit Distance & Confusion Matrix</h2>', 
                unsafe_allow_html=True)
    
    if 'weighted_checker' not in models:
        st.error("Weighted spell checker not found. Please run Extra Task first.")
        return
    
    weighted_checker = models['weighted_checker']
    confusion_matrix = weighted_checker.confusion_matrix
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Vocabulary Size", f"{len(weighted_checker.vocabulary):,}")
    with col2:
        st.metric("Confusion Patterns Learned", f"{confusion_matrix.total_errors}")
    
    # Show confusion matrix
    st.markdown("### Character Confusion Matrix")
    st.write("Most common character confusions (char1 → char2):")
    
    all_confusions = []
    for char1, char2_dict in confusion_matrix.confusion_counts.items():
        for char2, count in char2_dict.items():
            if count > 0:
                all_confusions.append((char1, char2, count))
    
    all_confusions.sort(key=lambda x: x[2], reverse=True)
    
    conf_df = pd.DataFrame([
        {
            'From': c1 if c1 else '<empty>',
            'To': c2 if c2 else '<empty>',
            'Count': count,
            'Weight': f"{confusion_matrix.get_confusion_weight(c1, c2):.3f}"
        }
        for c1, c2, count in all_confusions[:20]
    ])
    st.dataframe(conf_df, use_container_width=True, hide_index=True)
    
    # Test weighted spell checking
    st.markdown("### Test Weighted Spell Checker")
    
    test_word = st.text_input("Enter a word to check (weighted):",
                              value="azarbaycan")
    
    if st.button("Check with Weighted Distance", key="weighted_check_btn"):
        suggestions = weighted_checker.correct_word(test_word, max_distance=2.0, top_n=5)
        
        if suggestions:
            st.write("**Suggestions (weighted by confusion probabilities):**")
            
            sugg_df = pd.DataFrame([
                {
                    'Suggestion': sugg,
                    'Weighted Distance': f"{dist:.3f}",
                    'Frequency': freq
                }
                for sugg, dist, freq in suggestions
            ])
            st.dataframe(sugg_df, use_container_width=True, hide_index=True)
            
            # Compare with regular Levenshtein
            if 'spell_checker' in models:
                st.markdown("#### Comparison with Regular Levenshtein")
                regular_suggestions = models['spell_checker'].correct_word(test_word, max_distance=2, top_n=5)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Weighted (Character-aware):**")
                    st.write([s[0] for s in suggestions[:3]])
                with col2:
                    st.write("**Regular (Uniform weights):**")
                    st.write([s[0] for s in regular_suggestions[:3]])
        else:
            st.error("No suggestions found.")


def main():
    """Main Streamlit app"""
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select a page:",
        ["Overview", "Task 1: Tokenization", "Task 2: Heaps' Law", 
         "Task 3: BPE", "Task 4: Segmentation", "Task 5: Spell Checker",
         "Extra: Weighted Edit Distance"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This tool implements various NLP tasks for Azerbaijani text processing. "
        "Run `python main.py --sample` to process data first."
    )
    
    # Load models
    output_dir = './outputs'
    if not os.path.exists(output_dir):
        st.error(f"Output directory '{output_dir}' not found. Please run the main pipeline first.")
        st.code("python main.py --sample --output ./outputs")
        return
    
    models = load_models(output_dir)
    
    # Show selected page
    if page == "Overview":
        show_overview(models)
    elif page == "Task 1: Tokenization":
        show_task1(models)
    elif page == "Task 2: Heaps' Law":
        show_task2(models)
    elif page == "Task 3: BPE":
        show_task3(models)
    elif page == "Task 4: Segmentation":
        show_task4(models)
    elif page == "Task 5: Spell Checker":
        show_task5(models)
    elif page == "Extra: Weighted Edit Distance":
        show_extra_task(models)


if __name__ == "__main__":
    main()
