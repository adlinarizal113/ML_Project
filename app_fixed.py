                    else:
                        st.error(
                            f"Unsupported input type: {name}"
                        )
                        st.stop()

                submitted = st.form_submit_button(
                    "Predict Severity"
                )

            if submitted:

                # Load only a trusted model from the team.
                pipeline = joblib.load(MODEL)

                sample = pd.DataFrame(
                    [values],
                    columns=[
                        item["name"]
                        for item in features
                    ]
                )

                prediction = pipeline.predict(sample)[0]

                st.success(
                    f"Predicted Accident Severity: {prediction}"
                )

                if hasattr(pipeline, "predict_proba"):

                    probabilities = pipeline.predict_proba(
                        sample
                    )[0]

                    labels = getattr(
                        pipeline,
                        "classes_",
                        None
                    )

                    if labels is not None:

                        st.subheader(
                            "Prediction Probabilities"
                        )

                        probability_data = pd.Series(
                            probabilities,
                            index=[
                                str(label)
                                for label in labels
                            ]
                        )

                        st.bar_chart(probability_data)

                st.caption(
                    "Educational prediction only. "
                    "Not an official road safety assessment."
                )

        except Exception as error:

            st.error(
                f"Model integration error: {error}"
            )

            st.info(
                "Please ask Nurin to verify the model, "
                "input features and package versions."
            )


# =====================================
# 6. ABOUT PROJECT
# =====================================

else:

    st.header("ℹ️ About This Project")

    st.subheader("Project Title")

    st.write(
        "Road Accident Severity Prediction "
        "Using Machine Learning in Pahang, Malaysia"
    )

    st.subheader("Project Objective")

    st.write(
        "This project aims to develop a machine learning "
        "system to classify road accident severity using "
        "Malaysian MIROS accident data."
    )

    st.subheader("Main Dataset")

    st.write(
        "MIROS-UMP DAX Challenge 2026 Dataset"
    )

    st.subheader("Target Variable")

    st.write("Type of Crash (Proposed)")

    st.subheader("Project Team")

    st.write("Project Leader: Gabriella")
    st.write("Data Engineer: Gabriellyn")
    st.write("Data Analyst: Zul")
    st.write("ML Engineer: Nurin")
    st.write("System Developer: Adlina")

    st.divider()

    st.info(
        "This system classifies accident severity "
        "within the scope of the training data. "
        "It does not predict whether an accident "
        "will occur."
    )
