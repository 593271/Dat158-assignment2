from flask_wtf import FlaskForm
from wtforms import StringField, IntegerField, FloatField, SelectField, SelectField, RadioField, BooleanField, SubmitField
from wtforms.validators import DataRequired, NumberRange


class DataForm(FlaskForm):

    """
    The form for entering values during patient encounter. Feel free to add additional 
    fields for the remaining features in the data set (features missing in the form 
    are set to default values in `predict.py`).
    """

    overallqual = IntegerField('Rate the overall quality of your house from 1-10 (1 being the worst and 10 being the best)', validators=[DataRequired()])
    yearbuilt = IntegerField('What year was the house built?')

    gla = FloatField('What is the gla (gross living area) of your property?')
    sodium = FloatField('Average sodium level during encounter')


    asthma = BooleanField(label='Pool')
    garage = BooleanField(label='Garage')
    depress = BooleanField(label='Basement')
    malnutrition = BooleanField(label='Fireplace')

    submit = SubmitField('Submit')

