/*
 * This file is part of Adblock Plus <https://adblockplus.org/>,
 * Copyright (C) 2006-present eyeo GmbH
 *
 * Adblock Plus is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * Adblock Plus is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Adblock Plus.  If not, see <http://www.gnu.org/licenses/>.
 */

"use strict";

/**
 * @fileOverview Matcher class implementing matching addresses against
 *               a list of filters.
 */

const {Filter, WhitelistFilter} = require("./filterClasses");

/**
 * Blacklist/whitelist filter matching
 * @constructor
 */
function Matcher()
{
  this.clear();
}
exports.Matcher = Matcher;

Matcher.prototype = {
  model: null,
  /**
   * Lookup table for filters by their associated keyword
   * @type {Map.<string,(Filter|Filter[])>}
   */
  filterByKeyword: null,

  /**
   * Lookup table for keywords by the filter
   * @type {Map.<Filter,string>}
   */
  keywordByFilter: null,

  /**
   * Removes all known filters
   */
  clear()
  {
    this.filterByKeyword = new Map();
    this.keywordByFilter = new Map();
  },

  /**
   * Adds a filter to the matcher
   * @param {RegExpFilter} filter
   */
  add(filter)
  {
    if (this.keywordByFilter.has(filter))
      return;

    // Look for a suitable keyword
    let keyword = this.findKeyword(filter);
    let oldEntry = this.filterByKeyword.get(keyword);
    if (typeof oldEntry == "undefined")
      this.filterByKeyword.set(keyword, filter);
    else if (oldEntry.length == 1)
      this.filterByKeyword.set(keyword, [oldEntry, filter]);
    else
      oldEntry.push(filter);
    this.keywordByFilter.set(filter, keyword);
  },

  /**
   * Removes a filter from the matcher
   * @param {RegExpFilter} filter
   */
  remove(filter)
  {
    let keyword = this.keywordByFilter.get(filter);
    if (typeof keyword == "undefined")
      return;

    let list = this.filterByKeyword.get(keyword);
    if (list.length <= 1)
      this.filterByKeyword.delete(keyword);
    else
    {
      let index = list.indexOf(filter);
      if (index >= 0)
      {
        list.splice(index, 1);
        if (list.length == 1)
          this.filterByKeyword.set(keyword, list[0]);
      }
    }

    this.keywordByFilter.delete(filter);
  },

  /**
   * Chooses a keyword to be associated with the filter
   * @param {Filter} filter
   * @return {string} keyword or an empty string if no keyword could be found
   */
  findKeyword(filter)
  {
    let result = "";
    let {text} = filter;
    if (Filter.regexpRegExp.test(text))
      return result;

    // Remove options
    let match = Filter.optionsRegExp.exec(text);
    if (match)
      text = match.input.substr(0, match.index);

    // Remove whitelist marker
    if (text[0] == "@" && text[1] == "@")
      text = text.substr(2);

    let candidates = text.toLowerCase().match(
      /[^a-z0-9%*][a-z0-9%]{3,}(?=[^a-z0-9%*])/g
    );
    if (!candidates)
      return result;

    let hash = this.filterByKeyword;
    let resultCount = 0xFFFFFF;
    let resultLength = 0;
    for (let i = 0, l = candidates.length; i < l; i++)
    {
      let candidate = candidates[i].substr(1);
      let filters = hash.get(candidate);
      let count = typeof filters != "undefined" ? filters.length : 0;
      if (count < resultCount ||
          (count == resultCount && candidate.length > resultLength))
      {
        result = candidate;
        resultCount = count;
        resultLength = candidate.length;
      }
    }
    return result;
  },

  /**
   * Checks whether a particular filter is being matched against.
   * @param {RegExpFilter} filter
   * @return {boolean}
   */
  hasFilter(filter)
  {
    return this.keywordByFilter.has(filter);
  },

  /**
   * Returns the keyword used for a filter, null for unknown filters.
   * @param {RegExpFilter} filter
   * @return {?string}
   */
  getKeywordForFilter(filter)
  {
    let keyword = this.keywordByFilter.get(filter);
    return typeof keyword != "undefined" ? keyword : null;
  },

  /**
   * Checks whether the entries for a particular keyword match a URL
   * @param {string} keyword
   * @param {string} location
   * @param {number} typeMask
   * @param {string} docDomain
   * @param {boolean} thirdParty
   * @param {string} sitekey
   * @param {boolean} specificOnly
   * @return {?Filter}
   */
  _checkEntryMatch(keyword, location, typeMask, docDomain, thirdParty, sitekey,
                   specificOnly)
  {
    let list = this.filterByKeyword.get(keyword);
    if (typeof list == "undefined")
      return null;
    for (let i = 0; i < list.length; i++)
    {
      let filter = list[i];

      if (specificOnly && filter.isGeneric() &&
          !(filter instanceof WhitelistFilter))
        continue;

      if (filter.matches(location, typeMask, docDomain, thirdParty, sitekey))
        return filter;
    }
    return null;
  },

  /**
   * Tests whether the URL matches any of the known filters
   * @param {string} location
   *   URL to be tested
   * @param {number} typeMask
   *   bitmask of content / request types to match
   * @param {string} docDomain
   *   domain name of the document that loads the URL
   * @param {boolean} thirdParty
   *   should be true if the URL is a third-party request
   * @param {string} sitekey
   *   public key provided by the document
   * @param {boolean} specificOnly
   *   should be true if generic matches should be ignored
   * @return {?RegExpFilter}
   *   matching filter or null
   */
  matchesAny(location, typeMask, docDomain, thirdParty, sitekey, specificOnly)
  {
    let candidates = location.toLowerCase().match(/[a-z0-9%]{3,}/g);
    if (candidates === null)
      candidates = [];
    candidates.push("");
    let result = null;
    for (let i = 0, l = candidates.length; i < l; i++)
    {
      result = this._checkEntryMatch(candidates[i], location, typeMask,
                                         docDomain, thirdParty, sitekey,
                                         specificOnly);
      if (result)
      {
        break;
      }
    }

    return result;
  }
};

/**
 * Combines a matcher for blocking and exception rules, automatically sorts
 * rules into two Matcher instances.
 * @constructor
 * @augments Matcher
 */
function CombinedMatcher()
{
  this.blacklist = new Matcher();
  this.whitelist = new Matcher();
  this.resultCache = new Map();
}
exports.CombinedMatcher = CombinedMatcher;

/**
 * Maximal number of matching cache entries to be kept
 * @type {number}
 */
CombinedMatcher.maxCacheEntries = 1000;

CombinedMatcher.prototype =
{
  /**
   * Matcher for blocking rules.
   * @type {Matcher}
   */
  blacklist: null,

  /**
   * Matcher for exception rules.
   * @type {Matcher}
   */
  whitelist: null,

  /**
   * Lookup table of previous matchesAny results
   * @type {Map.<string,Filter>}
   */
  resultCache: null,

  /**
   * @see Matcher#clear
   */
  clear()
  {
    this.blacklist.clear();
    this.whitelist.clear();
    this.resultCache.clear();
  },

  /**
   * @see Matcher#add
   * @param {Filter} filter
   */
  add(filter)
  {
    if (filter instanceof WhitelistFilter)
      this.whitelist.add(filter);
    else
      this.blacklist.add(filter);

    this.resultCache.clear();
  },

  /**
   * @see Matcher#remove
   * @param {Filter} filter
   */
  remove(filter)
  {
    if (filter instanceof WhitelistFilter)
      this.whitelist.remove(filter);
    else
      this.blacklist.remove(filter);

    this.resultCache.clear();
  },

  /**
   * @see Matcher#findKeyword
   * @param {Filter} filter
   * @return {string} keyword
   */
  findKeyword(filter)
  {
    if (filter instanceof WhitelistFilter)
      return this.whitelist.findKeyword(filter);
    return this.blacklist.findKeyword(filter);
  },

  /**
   * @see Matcher#hasFilter
   * @param {Filter} filter
   * @return {boolean}
   */
  hasFilter(filter)
  {
    if (filter instanceof WhitelistFilter)
      return this.whitelist.hasFilter(filter);
    return this.blacklist.hasFilter(filter);
  },

  /**
   * @see Matcher#getKeywordForFilter
   * @param {Filter} filter
   * @return {string} keyword
   */
  getKeywordForFilter(filter)
  {
    if (filter instanceof WhitelistFilter)
      return this.whitelist.getKeywordForFilter(filter);
    return this.blacklist.getKeywordForFilter(filter);
  },

  /**
   * Checks whether a particular filter is slow
   * @param {RegExpFilter} filter
   * @return {boolean}
   */
  isSlowFilter(filter)
  {
    let matcher = (
      filter instanceof WhitelistFilter ? this.whitelist : this.blacklist
    );
    if (matcher.hasFilter(filter))
      return !matcher.getKeywordForFilter(filter);
    return !matcher.findKeyword(filter);
  },

  /**
   * Optimized filter matching testing both whitelist and blacklist matchers
   * simultaneously. For parameters see
     {@link Matcher#matchesAny Matcher.matchesAny()}.
   * @see Matcher#matchesAny
   * @inheritdoc
   */
  matchesAnyInternal(location, typeMask, docDomain, thirdParty, sitekey,
                     specificOnly)
  {
    let candidates = location.toLowerCase().match(/[a-z0-9%]{3,}/g);
    if (candidates === null)
      candidates = [];
    candidates.push("");

    let blacklistHit = null;
    for (let i = 0, l = candidates.length; i < l; i++)
    {
      let substr = candidates[i];
      let result = this.whitelist._checkEntryMatch(
        substr, location, typeMask, docDomain, thirdParty, sitekey
      );
      if (result)
        return result;
      if (blacklistHit === null)
      {
        blacklistHit = this.blacklist._checkEntryMatch(
          substr, location, typeMask, docDomain, thirdParty, sitekey,
          specificOnly
        );
      }
    }
    return blacklistHit;
  },

  /**
   * @see Matcher#matchesAny
   * @inheritdoc
   */
  matchesAny(location, typeMask, docDomain, thirdParty, sitekey, specificOnly)
  {
    let key = location + " " + typeMask + " " + docDomain + " " + thirdParty +
      " " + sitekey + " " + specificOnly;

    let result = this.resultCache.get(key);
    if (typeof result != "undefined")
      return result;

    result = this.matchesAnyInternal(location, typeMask, docDomain,
                                     thirdParty, sitekey, specificOnly);

    if (this.resultCache.size >= CombinedMatcher.maxCacheEntries)
      this.resultCache.clear();

    this.resultCache.set(key, result);

    return result;
  }
};

/**
 * Shared CombinedMatcher instance that should usually be used.
 * @type {CombinedMatcher}
 */
exports.defaultMatcher = new CombinedMatcher();
