*** Settings ***
Library    SeleniumLibrary

*** Variables ***
${Url}           http://app:8000
${REMOTE_URL}    http://selenium:4444/wd/hub

*** Test Cases ***
Valid Input
    Open Browser To Initial Page
    Type In Recommendation  "[101, 102]"
    Submit Recommendation
    Recommendation Page Should Be Open
    [Teardown]    Close Browser

*** Keywords ***
Open Browser To Initial Page
    ${options}=    Evaluate    sys.modules['selenium.webdriver'].ChromeOptions()    sys, selenium.webdriver
    Call Method    ${options}    add_argument    --headless
    Call Method    ${options}    add_argument    --no-sandbox
    Create WebDriver    Remote    command_executor=${REMOTE_URL}    options=${options}
    Go To    ${Url}
    Title Should Be    Recommendation Form

Type In Recommendation
    [Arguments]    ${username}
    Input Text    id=recommendation    ${username}

Submit Recommendation
    Click Button    id=submit_ids

Recommendation Page Should Be Open
    Wait Until Page Contains Element    id=recommendations-output    timeout=60s

# pip install selenium
# pip install robotframework
# pip install robotframework-seleniumlibrary
