import streamlit as st
import pandas as pd
from recommendations.smart_recommender import SmartRecommender
from data.create_sample_data import create_sample_data
import traceback

def init_session_state():
    """Initialize session state variables"""
    try:
        if 'recommender' not in st.session_state:
            with st.spinner('Initializing recommender system...'):
                st.session_state.recommender = SmartRecommender()
                sample_data = create_sample_data()
                st.session_state.recommender.update_damage_patterns(sample_data)
    except Exception as e:
        st.error(f"Failed to initialize recommender: {str(e)}")
        st.stop()

def main():
    try:
        st.set_page_config(
            page_title="Insurance Claims Dashboard",
            page_icon="🚗",
            layout="wide"
        )

        st.title("Insurance Claims Smart Recommender")
        
        # Initialize session state
        init_session_state()
        
        # Sidebar for input
        with st.sidebar:
            st.header("Damage Assessment Input")
            
            part_code = st.text_input("Part Code", "1001")
            damage_severity = st.selectbox(
                "Damage Severity",
                ["minor", "moderate", "severe", "critical"]
            )
            
            process_button = st.button("Get Recommendations")

        if process_button:
            with st.spinner('Generating recommendations...'):
                recommendations = st.session_state.recommender.get_recommendations(
                    part_code, 
                    damage_severity
                )
                
                if recommendations:
                    # Display recommendations in tabs
                    tab1, tab2 = st.tabs(["Parts & Repairs", "Costs & Priority"])
                    
                    with tab1:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Associated Parts")
                            if recommendations['associated_parts']:
                                for part in recommendations['associated_parts']:
                                    st.info(
                                        f"Part: {part['part_code']}\n"
                                        f"Confidence: {part['confidence']:.2f}%"
                                    )
                            else:
                                st.info("No associated parts found")
                        
                        with col2:
                            st.subheader("Repair Suggestions")
                            for suggestion in recommendations['repair_suggestions']:
                                st.success(f"• {suggestion}")
                    
                    with tab2:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Cost Estimates")
                            cost_data = recommendations['cost_estimates']
                            st.metric(
                                "Estimated Cost",
                                f"${cost_data['estimated_cost']:.2f}",
                                delta=None
                            )
                            st.write(f"Confidence Level: {cost_data['confidence_level']}")
                            st.write(f"Includes Labor: {cost_data['includes_labor']}")
                        
                        with col2:
                            st.subheader("Priority Level")
                            priority = recommendations['priority_level']
                            if priority['requires_immediate_action']:
                                st.error(f"Priority Level: {priority['level'].upper()}")
                            else:
                                st.info(f"Priority Level: {priority['level'].upper()}")
                            st.write(f"Score: {priority['score']}/4")
                else:
                    st.error("Failed to generate recommendations")

    except Exception as e:
        st.error("An error occurred in the application")
        st.error(f"Error details: {str(e)}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main() 