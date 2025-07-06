from enum import Enum

class Conference(str, Enum):
    EASTERN = "Eastern"
    WESTERN = "Western"

class Division(str, Enum):
    ATLANTIC = "Atlantic"
    CENTRAL = "Central"
    SOUTHEAST = "Southeast"
    NORTHWEST = "Northwest"
    PACIFIC = "Pacific"
    SOUTHWEST = "Southwest"

    # WNBA Specific
    EASTERN_CONFERENCE = "Eastern Conference"
    WESTERN_CONFERENCE = "Western Conference" 